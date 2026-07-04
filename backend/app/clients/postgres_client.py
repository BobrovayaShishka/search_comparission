import logging
from typing import Any

import asyncpg
from asyncpg import Pool

from app.config import Settings

logger = logging.getLogger(__name__)


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

    async def connect(self) -> None:
        """Create connection pool with configured limits and timeouts."""
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            dsn=self._settings.database_url,
            min_size=self._settings.pg_pool_min_size,
            max_size=self._settings.pg_pool_max_size,
            command_timeout=self._settings.pg_command_timeout,
        )
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

    async def upsert_product_ref(
        self,
        tenant_id: str,
        qdrant_point_id: str,
        sku: str | None,
    ) -> None:
        """Store tenant ↔ Qdrant point linkage (no content duplication)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO product_refs (tenant_id, qdrant_point_id, sku)
                VALUES ($1::uuid, $2::uuid, $3)
                ON CONFLICT (qdrant_point_id) DO UPDATE SET sku = EXCLUDED.sku, updated_at = NOW()
                """,
                tenant_id,
                qdrant_point_id,
                sku,
            )

    async def delete_product_ref(self, qdrant_point_id: str) -> None:
        """Remove product reference after hard delete."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM product_refs WHERE qdrant_point_id = $1::uuid",
                qdrant_point_id,
            )
