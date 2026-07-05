import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import Settings

logger = logging.getLogger(__name__)

# AsyncQdrantClient(...) само по себе не подключается (ленивая инициализация) —
# поэтому ретраим не конструктор, а первый реальный вызов (get_collections),
# который и обнаружит "Qdrant ещё не поднялся" в docker-compose/k8s на старте.
_connect_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class QdrantManager:
    """Manages AsyncQdrantClient with gRPC preference for low-latency search."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("Qdrant client is not initialized")
        return self._client

    @property
    def collection(self) -> str:
        return self._settings.qdrant_collection

    @_connect_retry
    async def _verify_connectivity(self, client: AsyncQdrantClient) -> None:
        await client.get_collections()

    async def connect(self) -> None:
        """Initialize gRPC-enabled async client and verify it can actually reach Qdrant."""
        if self._client is not None:
            return

        client = AsyncQdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_http_port,
            grpc_port=self._settings.qdrant_grpc_port,
            prefer_grpc=self._settings.qdrant_prefer_grpc,
            timeout=self._settings.network_timeout_seconds,
        )
        await self._verify_connectivity(client)
        self._client = client
        logger.info(
            "Qdrant client ready (host=%s, grpc=%d, prefer_grpc=%s)",
            self._settings.qdrant_host,
            self._settings.qdrant_grpc_port,
            self._settings.qdrant_prefer_grpc,
        )

    async def close(self) -> None:
        """Close client connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client closed")

    async def health_check(self) -> dict[str, Any]:
        """Check Qdrant connectivity and collection status."""
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        collection_info: dict[str, Any] | None = None

        if self.collection in names:
            info = await self.client.get_collection(self.collection)
            current_m = info.hnsw_config.m if info.hnsw_config else None
            collection_info = {
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value if info.status else "unknown",
                "optimizer_status": info.optimizer_status.status.value
                if info.optimizer_status
                else "unknown",
                "hnsw_m": current_m,
                # m=0 outside of an active bulk load means indexing was
                # disabled and never re-enabled — usually a bulk_load() that
                # crashed before its finally-block ran, or is still in flight.
                "indexing_disabled": current_m == 0,
            }

        return {
            "status": "ok",
            "collections": names,
            "target_collection": self.collection,
            "collection_info": collection_info,
        }

    async def check_payload_indexes(self) -> list[dict[str, Any]]:
        """Return payload index configuration for the target collection."""
        if self.collection not in [
            c.name for c in (await self.client.get_collections()).collections
        ]:
            return []

        info = await self.client.get_collection(self.collection)
        payload_schema = info.payload_schema or {}
        return [
            {"field": field, "schema": schema.model_dump() if hasattr(schema, "model_dump") else schema}
            for field, schema in payload_schema.items()
        ]
