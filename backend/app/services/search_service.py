import asyncio
import logging
import time
from typing import Any

from qdrant_client.http import models

from app.cache.search_cache import SearchCache
from app.clients.postgres_client import PostgresManager
from app.clients.qdrant_client import QdrantManager
from app.config import Settings
from app.exceptions import UnindexedFilterFieldError
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
        # Держим ссылки на фоновые таски (fire-and-forget логирование поиска
        # в Postgres), иначе asyncio может собрать их GC до завершения, а при
        # graceful shutdown они просто обрываются на середине без следа.
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain_background_tasks(self, timeout: float = 5.0) -> None:
        """Wait for in-flight background tasks (e.g. search logging) to finish before shutdown."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        logger.info("Draining %d background task(s) before shutdown", len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in still_pending:
            logger.warning("Background task did not finish before shutdown timeout — cancelling")
            task.cancel()

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
            if key not in self._settings.metadata_indexed_fields:
                # Отказ, а не молчаливый fallback на неиндексированный скан:
                # неиндексированный payload-фильтр на 10М+ векторов — это
                # гарантированная деградация latency без предупреждения.
                raise UnindexedFilterFieldError(key, self._settings.metadata_indexed_fields)
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
            # AsyncQdrantClient.search() устарел в клиентах 1.10+ в пользу
            # query_points() (унифицированный query-API: search/recommend/
            # discover через один метод). Возвращает QueryResponse с полем
            # .points вместо голого списка ScoredPoint.
            response_qdrant = await self._qdrant.client.query_points(
                collection_name=self._qdrant.collection,
                query=request.embedding,
                query_filter=query_filter,
                limit=limit,
                offset=request.offset,
                score_threshold=request.score_threshold,
                search_params=models.SearchParams(hnsw_ef=hnsw_ef),
                with_payload=True,
            )
            results = response_qdrant.points
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

        self._spawn_background(
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
