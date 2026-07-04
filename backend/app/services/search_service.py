import asyncio
import logging
import time
from typing import Any

from qdrant_client.http import models

from app.cache.search_cache import SearchCache
from app.clients.postgres_client import PostgresManager
from app.clients.qdrant_client import QdrantManager
from app.config import Settings
from app.models.schemas import (
    ProductPayload,
    SearchFilters,
    SearchHit,
    SearchResponse,
    VectorSearchRequest,
)
from app.monitoring.metrics import hash_query, search_metrics

logger = logging.getLogger(__name__)


class SearchService:
    """
    Primary vector search — Qdrant-only reads, no PostgreSQL content fetch.

    Uses gRPC, HNSW ef search, payload filters, and result caching.
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        postgres: PostgresManager,
        cache: SearchCache,
        settings: Settings,
    ) -> None:
        self._qdrant = qdrant
        self._postgres = postgres
        self._cache = cache
        self._settings = settings

    def _build_filter(self, filters: SearchFilters) -> models.Filter | None:
        """Build indexed Qdrant filter from search parameters."""
        conditions: list[models.Condition] = []

        if filters.tenant_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=filters.tenant_id),
                )
            )
        if filters.category is not None:
            conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=filters.category),
                )
            )
        if filters.is_active is not None:
            conditions.append(
                models.FieldCondition(
                    key="is_active",
                    match=models.MatchValue(value=filters.is_active),
                )
            )
        if filters.is_deleted is not None:
            conditions.append(
                models.FieldCondition(
                    key="is_deleted",
                    match=models.MatchValue(value=filters.is_deleted),
                )
            )
        for key, value in filters.metadata.items():
            conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )

        if not conditions:
            return None
        return models.Filter(must=conditions)

    async def search(self, request: VectorSearchRequest) -> SearchResponse:
        """
        Execute vector similarity search with optional filters and pagination.

        Results include full payload from Qdrant — zero PostgreSQL reads for content.
        """
        start = time.perf_counter()
        hnsw_ef = request.hnsw_ef or self._settings.search_hnsw_ef
        limit = min(request.limit, self._settings.search_max_limit)

        cache_key = hash_query(
            request.embedding,
            request.filters.model_dump(exclude_none=True),
            limit,
            request.offset,
        )
        cached = await self._cache.get(cache_key)
        if cached is not None:
            latency = (time.perf_counter() - start) * 1000
            search_metrics.record_request(latency, cache_hit=True)
            return SearchResponse(**cached, cache_hit=True, latency_ms=round(latency, 2))

        query_filter = self._build_filter(request.filters)

        try:
            results = await self._qdrant.client.search(
                collection_name=self._qdrant.collection,
                query_vector=request.embedding,
                query_filter=query_filter,
                limit=limit,
                offset=request.offset,
                score_threshold=request.score_threshold,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
                with_payload=True,
            )
        except Exception:
            search_metrics.record_error()
            raise

        hits = [
            SearchHit(
                id=str(point.id),
                score=point.score,
                payload=ProductPayload(**point.payload),
            )
            for point in results
            if point.payload
        ]

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response = SearchResponse(
            hits=hits,
            total=len(hits),
            limit=limit,
            offset=request.offset,
            latency_ms=latency_ms,
            cache_hit=False,
        )

        await self._cache.set(cache_key, response.model_dump())
        search_metrics.record_request(latency_ms)

        asyncio.create_task(
            self._postgres.log_search(
                tenant_id=request.filters.tenant_id,
                query_hash=cache_key[:16],
                latency_ms=latency_ms,
                results_count=len(hits),
                cache_hit=False,
                filters=request.filters.model_dump(exclude_none=True),
            )
        )

        return response

    async def scroll(
        self,
        filters: SearchFilters,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Offset-based pagination over filtered points (non-vector browse).

        Efficient for listing products within a tenant/category.
        """
        query_filter = self._build_filter(filters)
        records, _ = await self._qdrant.client.scroll(
            collection_name=self._qdrant.collection,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        items = [
            {"id": str(r.id), "payload": r.payload}
            for r in records
            if r.payload
        ]
        return {"items": items, "limit": limit, "offset": offset, "count": len(items)}
