import asyncio
import logging
from typing import Any

import asyncpg
from asyncpg import Pool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import Settings

logger = logging.getLogger(__name__)

# Транзиентные ошибки, на которые имеет смысл ретраить (сеть/пул/deadlock).
# Ошибки констрейнтов (unique violation и т.п.) НЕ ретраим — они не самоисправятся.
_RETRYABLE_ERRORS = (
    asyncpg.PostgresConnectionError,
    asyncpg.TooManyConnectionsError,
    asyncpg.DeadlockDetectedError,
    asyncpg.ConnectionDoesNotExistError,
    TimeoutError,
    OSError,
)

# Ретрай ТОЛЬКО на старте контейнера/подключении к пулу: Postgres может быть
# ещё не готов принимать соединения (типичная ситуация в docker-compose/k8s,
# где порядок старта сервисов не гарантирован). 5 попыток с экспоненциальным
# backoff — от 0.5с до ~10с между попытками, суммарно до ~30с ожидания.
_connect_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, max=10),
    retry=retry_if_exception_type((OSError, asyncpg.PostgresConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class PostgresManager:
    """Manages asyncpg connection pool for operational PostgreSQL data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Pool | None = None

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        return self._pool

    @_connect_retry
    async def _create_pool(self) -> Pool:
        return await asyncpg.create_pool(
            dsn=self._settings.database_url,
            min_size=self._settings.pg_pool_min_size,
            max_size=self._settings.pg_pool_max_size,
            command_timeout=self._settings.pg_command_timeout,
        )

    async def connect(self) -> None:
        """Create connection pool with configured limits and timeouts, retrying on startup races."""
        if self._pool is not None:
            return

        self._pool = await self._create_pool()
        logger.info(
            "PostgreSQL pool ready (min=%d, max=%d)",
            self._settings.pg_pool_min_size,
            self._settings.pg_pool_max_size,
        )

    async def close(self) -> None:
        """Gracefully close all pool connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    async def health_check(self) -> dict[str, Any]:
        """Verify pool connectivity and return server version."""
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            tables = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
        return {"status": "ok", "version": version, "tables": tables}

    async def _with_retry(self, fn, *, retries: int = 3, base_delay: float = 0.2):
        """
        Run an async DB operation with exponential backoff on transient errors.

        Constraint violations, check violations, etc. are raised immediately —
        retrying a logically-invalid write only wastes time.
        """
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await fn()
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Transient Postgres error (attempt %d/%d): %s — retrying in %.2fs",
                    attempt + 1, retries, exc, delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def log_search(
        self,
        tenant_id: str | None,
        query_hash: str,
        latency_ms: float,
        results_count: int,
        cache_hit: bool,
        filters: dict[str, Any],
    ) -> None:
        """Persist search telemetry without blocking the hot path."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO search_logs (tenant_id, query_hash, latency_ms, results_count, cache_hit, filters)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb)
                    """,
                    tenant_id,
                    query_hash,
                    latency_ms,
                    results_count,
                    cache_hit,
                    filters,
                )
        except Exception:
            logger.exception("Failed to write search log")

    # ------------------------------------------------------------------
    # Dual-write support: product_refs is the "commit record" for the
    # Qdrant write. Pattern used by ProductService:
    #   1. insert_pending_ref()      — cheap, transactional, safe to abort
    #   2. <write content to Qdrant> — the actual source of truth
    #   3. mark_ref_active()         — confirms the pair is consistent
    # If step 2 fails: delete_product_ref() compensates (removes the pending row).
    # If step 3 fails after retries: row stays 'pending' and is picked up
    # later by ReconciliationService, which checks whether the Qdrant point
    # actually exists and resolves the row accordingly.
    # ------------------------------------------------------------------

    async def insert_pending_ref(
        self,
        tenant_id: str,
        qdrant_point_id: str,
        sku: str | None,
    ) -> None:
        """Create ref row in 'pending' state, before the Qdrant write happens."""
        async def _op():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO product_refs (tenant_id, qdrant_point_id, sku, version, status)
                    VALUES ($1::uuid, $2::uuid, $3, 1, 'pending')
                    ON CONFLICT (qdrant_point_id) DO UPDATE
                        SET sku = EXCLUDED.sku, status = 'pending', updated_at = NOW()
                    """,
                    tenant_id,
                    qdrant_point_id,
                    sku,
                )
        await self._with_retry(_op)

    async def mark_ref_active(self, qdrant_point_id: str) -> None:
        """Confirm the Qdrant write succeeded — flip ref to 'active'."""
        async def _op():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE product_refs SET status = 'active', updated_at = NOW() WHERE qdrant_point_id = $1::uuid",
                    qdrant_point_id,
                )
        await self._with_retry(_op)

    async def upsert_product_ref(
        self,
        tenant_id: str,
        qdrant_point_id: str,
        sku: str | None,
    ) -> None:
        """Direct active upsert (used when the Qdrant side is already confirmed, e.g. sku-only changes)."""
        async def _op():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO product_refs (tenant_id, qdrant_point_id, sku, status)
                    VALUES ($1::uuid, $2::uuid, $3, 'active')
                    ON CONFLICT (qdrant_point_id) DO UPDATE
                        SET sku = EXCLUDED.sku, status = 'active', updated_at = NOW()
                    """,
                    tenant_id,
                    qdrant_point_id,
                    sku,
                )
        await self._with_retry(_op)

    async def delete_product_ref(self, qdrant_point_id: str) -> None:
        """Remove product reference (hard delete, or compensation after a failed Qdrant write)."""
        async def _op():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM product_refs WHERE qdrant_point_id = $1::uuid",
                    qdrant_point_id,
                )
        await self._with_retry(_op)

    # ------------------------------------------------------------------
    # Optimistic concurrency (compare-and-swap). Qdrant has no native
    # "update payload only if current value == X" primitive, so Postgres
    # plays lock/version-authority here even though Qdrant holds the content.
    # ------------------------------------------------------------------

    async def cas_bump_version(
        self,
        qdrant_point_id: str,
        expected_version: int,
        new_version: int,
    ) -> bool:
        """
        Atomically bump version only if it still matches expected_version.

        Returns True on success, False if another writer already changed it
        (caller should re-read the latest state and retry).
        """
        async def _op() -> bool:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE product_refs
                    SET version = $3, status = 'active', updated_at = NOW()
                    WHERE qdrant_point_id = $1::uuid AND version = $2
                    """,
                    qdrant_point_id,
                    expected_version,
                    new_version,
                )
            # asyncpg execute() returns e.g. "UPDATE 1" / "UPDATE 0"
            return result.endswith(" 1")
        return await self._with_retry(_op)

    async def revert_version(self, qdrant_point_id: str, back_to_version: int) -> None:
        """Best-effort compensation: revert a version bump if the Qdrant write that followed it failed."""
        async def _op():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE product_refs SET version = $2, updated_at = NOW() WHERE qdrant_point_id = $1::uuid",
                    qdrant_point_id,
                    back_to_version,
                )
        try:
            await self._with_retry(_op, retries=2)
        except Exception:
            logger.exception(
                "Failed to revert version for %s after failed Qdrant write — "
                "ref version may be ahead of actual content until next reconciliation run",
                qdrant_point_id,
            )

    # ------------------------------------------------------------------
    # Reconciliation support
    # ------------------------------------------------------------------

    async def find_stale_pending_refs(self, older_than_seconds: int = 300) -> list[dict[str, Any]]:
        """Refs stuck in 'pending' — either the Qdrant write never happened, or we crashed before confirming it."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT qdrant_point_id, tenant_id, sku, created_at
                FROM product_refs
                WHERE status = 'pending' AND created_at < NOW() - ($1 || ' seconds')::interval
                """,
                str(older_than_seconds),
            )
        return [dict(r) for r in rows]
