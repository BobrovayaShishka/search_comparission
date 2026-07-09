import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.config import Settings
from app.db import Database
from app.llm import LLMClient, TokenUsage, build_product_text

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings, db: Database, llm: LLMClient) -> None:
        self._settings = settings
        self._db = db
        self._llm = llm
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("Qdrant client is not initialized")
        return self._client

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncQdrantClient(url=self._settings.qdrant_url, timeout=30)
        await self._ensure_collection()
        logger.info("Qdrant connected")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _get_collection_dim(self) -> int | None:
        name = self._settings.qdrant_collection
        collections = await self.client.get_collections()
        if name not in [c.name for c in collections.collections]:
            return None
        info = await self.client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, models.VectorParams):
            return vectors.size
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            if isinstance(first, models.VectorParams):
                return first.size
        return None

    async def _recreate_collection(self) -> None:
        name = self._settings.qdrant_collection
        dim = self._settings.embedding_dimension
        collections = await self.client.get_collections()
        if name in [c.name for c in collections.collections]:
            await self.client.delete_collection(name)
        await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d)", name, dim)

    async def _ensure_collection(self) -> None:
        expected_dim = self._settings.embedding_dimension
        current_dim = await self._get_collection_dim()
        if current_dim is None:
            await self._recreate_collection()
            return
        if current_dim != expected_dim:
            logger.warning(
                "Qdrant dim mismatch: collection=%d, env=%d — recreating collection",
                current_dim,
                expected_dim,
            )
            await self._recreate_collection()

    async def health_check(self) -> dict:
        info = await self.client.get_collection(self._settings.qdrant_collection)
        dim = await self._get_collection_dim()
        return {
            "status": "ok",
            "points_count": info.points_count,
            "collection": self._settings.qdrant_collection,
            "vector_dim": dim,
            "expected_dim": self._settings.embedding_dimension,
        }

    def _load_products(self) -> list[dict]:
        path = Path(self._settings.products_json_path)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    async def seed_if_empty(self) -> dict:
        pg_count = await self._db.count_products()
        products = self._load_products()
        expected = len(products)

        if not products:
            return {"status": "empty", "count": 0}

        collection_dim = await self._get_collection_dim()
        expected_dim = self._settings.embedding_dimension
        dim_mismatch = collection_dim is not None and collection_dim != expected_dim

        qdrant_info = await self.client.get_collection(self._settings.qdrant_collection)
        points_count = qdrant_info.points_count

        needs_reseed = (
            dim_mismatch
            or pg_count < expected
            or points_count < expected
        )

        if not needs_reseed and pg_count > 0:
            return {"status": "skipped", "count": pg_count, "expected": expected}

        if dim_mismatch:
            logger.info(
                "Embedding dim change: %s → %d, re-seeding catalog",
                collection_dim,
                expected_dim,
            )
        elif pg_count > 0 and pg_count < expected:
            logger.info("Catalog upgrade: %d → %d products, re-seeding", pg_count, expected)
        elif points_count < expected:
            logger.info("Qdrant incomplete: %d → %d points, re-seeding", points_count, expected)

        await self._db.truncate_products()
        await self._recreate_collection()

        texts = [
            build_product_text(p["name"], p.get("description", ""), p.get("category", ""))
            for p in products
        ]
        embed_result = await self._llm.embed(texts)
        points: list[models.PointStruct] = []

        logger.info("Embedding %d products for seed...", len(products))

        for product, vector in zip(products, embed_result.vectors):
            await self._db.upsert_product(
                product_id=product["id"],
                name=product["name"],
                description=product.get("description", ""),
                category=product.get("category", ""),
                price=product.get("price"),
                sku=product.get("sku"),
            )
            points.append(
                models.PointStruct(
                    id=_to_uuid(product["id"]),
                    vector=vector,
                    payload={
                        "name": product["name"],
                        "description": product.get("description", ""),
                        "category": product.get("category", ""),
                        "price": product.get("price"),
                        "sku": product.get("sku"),
                    },
                )
            )

        batch = 32
        for i in range(0, len(points), batch):
            await self.client.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points[i : i + batch],
            )

        return {
            "status": "seeded",
            "count": len(products),
            "seed_embed_tokens": embed_result.tokens.embed_tokens,
        }

    async def search(self, query: str, limit: int) -> dict[str, Any]:
        start = time.perf_counter()
        limit = min(limit, self._settings.search_max_limit)

        embed_result = await self._llm.embed([query])
        vector = embed_result.vectors[0]

        qdrant_start = time.perf_counter()
        response = await self.client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        qdrant_ms = round((time.perf_counter() - qdrant_start) * 1000, 2)
        total_ms = round((time.perf_counter() - start) * 1000, 2)

        min_score = 0.0 if self._settings.mock_mode else self._settings.vector_min_score
        hits = []
        for point in response.points:
            if point.score is not None and point.score < min_score:
                continue
            payload = point.payload or {}
            hits.append({
                "id": str(point.id),
                "score": point.score,
                "name": payload.get("name", ""),
                "description": payload.get("description", ""),
                "category": payload.get("category", ""),
                "price": payload.get("price"),
                "sku": payload.get("sku"),
            })

        return {
            "query": query,
            "hits": hits,
            "total": len(hits),
            "min_score": min_score if min_score > 0 else None,
            "limit": limit,
            "latency_ms": total_ms,
            "latency_embed_ms": embed_result.latency_ms,
            "latency_qdrant_ms": qdrant_ms,
            "tokens": embed_result.tokens.to_dict(),
        }


def _to_uuid(value: str) -> str:
    return str(uuid.UUID(value))
