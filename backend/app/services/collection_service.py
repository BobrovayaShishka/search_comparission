import logging
from typing import Any

from qdrant_client.http import models

from app.clients.qdrant_client import QdrantManager
from app.config import Settings

logger = logging.getLogger(__name__)

PAYLOAD_INDEX_FIELDS: list[tuple[str, models.PayloadSchemaType]] = [
    ("tenant_id", models.PayloadSchemaType.KEYWORD),
    ("category", models.PayloadSchemaType.KEYWORD),
    ("is_active", models.PayloadSchemaType.BOOL),
    ("is_deleted", models.PayloadSchemaType.BOOL),
    ("sku", models.PayloadSchemaType.KEYWORD),
    ("created_at", models.PayloadSchemaType.DATETIME),
    ("updated_at", models.PayloadSchemaType.DATETIME),
    ("version", models.PayloadSchemaType.INTEGER),
]


class CollectionService:
    """Qdrant collection lifecycle: create, migrate, index management."""

    def __init__(self, qdrant: QdrantManager, settings: Settings) -> None:
        self._qdrant = qdrant
        self._settings = settings

    async def create_collection(
        self,
        *,
        recreate: bool = False,
        enable_quantization: bool = True,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
        disable_hnsw: bool = False,
    ) -> dict[str, Any]:
        """
        Create or recreate collection with production-optimized parameters.

        HNSW m=32, ef_construct=200, scalar INT8 quantization, on_disk_payload.
        Set disable_hnsw=True (m=0) during bulk import, then call finalize_indexes().
        """
        client = self._qdrant.client
        name = self._qdrant.collection
        dim = self._settings.embedding_dimension

        exists = name in [c.name for c in (await client.get_collections()).collections]
        if exists and recreate:
            logger.warning("Recreating collection %s", name)
            await client.delete_collection(name)
            exists = False

        if exists:
            logger.info("Collection %s already exists", name)
            await self.ensure_payload_indexes()
            return {"status": "exists", "collection": name}

        m = 0 if disable_hnsw else (hnsw_m or self._settings.qdrant_hnsw_m)
        ef_construct = hnsw_ef_construct or self._settings.qdrant_hnsw_ef_construct

        quantization_config = None
        if enable_quantization and not disable_hnsw:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        await client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
                on_disk=True,
            ),
            quantization_config=quantization_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000 if disable_hnsw else 10_000,
                memmap_threshold=50_000,
                default_segment_number=4,
            ),
            on_disk_payload=True,
        )

        await self.ensure_payload_indexes()
        logger.info("Collection %s created (m=%d, ef_construct=%d)", name, m, ef_construct)
        return {
            "status": "created",
            "collection": name,
            "hnsw_m": m,
            "hnsw_ef_construct": ef_construct,
            "quantization": enable_quantization,
        }

    async def ensure_payload_indexes(self) -> None:
        """Create payload indexes for all filterable fields."""
        client = self._qdrant.client
        name = self._qdrant.collection

        for field_name, schema_type in PAYLOAD_INDEX_FIELDS:
            try:
                await client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug("Payload index ensured: %s", field_name)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    logger.warning("Payload index %s: %s", field_name, exc)

    async def finalize_indexes(self) -> dict[str, Any]:
        """
        Re-enable HNSW indexing after bulk load (m=0 → production values).

        Triggers optimizer to build HNSW graph and enable quantization.
        """
        client = self._qdrant.client
        name = self._qdrant.collection

        await client.update_collection(
            collection_name=name,
            hnsw_config=models.HnswConfigDiff(
                m=self._settings.qdrant_hnsw_m,
                ef_construct=self._settings.qdrant_hnsw_ef_construct,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10_000,
            ),
        )

        logger.info("Collection %s indexes finalized", name)
        return {"status": "finalized", "hnsw_m": self._settings.qdrant_hnsw_m}

    async def disable_indexing_for_bulk(self) -> None:
        """Disable HNSW (m=0) and raise indexing threshold for fast bulk import."""
        await self._qdrant.client.update_collection(
            collection_name=self._qdrant.collection,
            hnsw_config=models.HnswConfigDiff(m=0),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        logger.info("Indexing disabled for bulk load")
