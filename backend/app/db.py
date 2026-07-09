import logging
import time
from typing import Any

import asyncpg
from asyncpg import Pool

from app.config import Settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Pool | None = None

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(dsn=self._settings.database_url, min_size=1, max_size=5)
        logger.info("PostgreSQL connected")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health_check(self) -> dict:
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM catalog_products")
        return {"status": "ok", "catalog_count": count}

    async def count_products(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM catalog_products")

    async def truncate_products(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE catalog_products")

    async def upsert_product(
        self,
        product_id: str,
        name: str,
        description: str,
        category: str,
        price: float | None,
        sku: str | None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO catalog_products (id, name, description, category, price, sku)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    category = EXCLUDED.category,
                    price = EXCLUDED.price,
                    sku = EXCLUDED.sku
                """,
                product_id,
                name,
                description,
                category,
                price,
                sku,
            )

    async def search_fulltext(self, query: str, limit: int) -> dict[str, Any]:
        start = time.perf_counter()
        limit = min(limit, self._settings.search_max_limit)

        rows = await self._fetch(query, limit, relaxed=False)
        mode = "strict"
        if not rows:
            rows = await self._fetch(query, limit, relaxed=True)
            mode = "relaxed"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        hits = [
            {
                "id": str(r["id"]),
                "score": float(r["rank"]),
                "name": r["name"],
                "description": r["description"],
                "category": r["category"],
                "price": float(r["price"]) if r["price"] is not None else None,
                "sku": r["sku"],
            }
            for r in rows
        ]
        return {
            "query": query,
            "hits": hits,
            "total": len(hits),
            "limit": limit,
            "mode": mode,
            "latency_ms": latency_ms,
            "tokens": None,
        }

    async def _fetch(self, query: str, limit: int, *, relaxed: bool) -> list[Any]:
        tsquery = "plainto_tsquery('russian', $1)" if relaxed else "websearch_to_tsquery('russian', $1)"
        sql = f"""
            SELECT id, name, description, category, price, sku,
                   ts_rank(search_vector, query) AS rank
            FROM catalog_products, {tsquery} query
            WHERE search_vector @@ query
            ORDER BY rank DESC
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, query, limit)
