import asyncio
import logging
import time
import uuid
from typing import Any

from qdrant_client.http import models

from app.clients.qdrant_client import QdrantManager
from app.config import Settings
from app.models.schemas import BulkLoadItem, BulkLoadStats
from app.monitoring.metrics import bulk_metrics
from app.services.collection_service import CollectionService

logger = logging.getLogger(__name__)


class BulkLoader:
    """
    High-throughput vector ingestion with parallel batches, retry, and rollback.

    Disables HNSW indexing during load, re-enables after completion.
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        collection_service: CollectionService,
        settings: Settings,
    ) -> None:
        self._qdrant = qdrant
        self._collection = collection_service
        self._settings = settings

    def _batch_size(self, override: int | None, vector_dim: int) -> int:
        if override:
            return min(max(override, 100), 500)
        if vector_dim >= 768:
            return 100
        if vector_dim >= 512:
            return 250
        return self._settings.bulk_batch_size

    async def _upsert_batch_with_retry(
        self,
        batch: list[BulkLoadItem],
        batch_idx: int,
    ) -> tuple[int, list[str], list[str]]:
        """Upsert a single batch with exponential backoff retry."""
        uploaded_ids: list[str] = []
        errors: list[str] = []
        client = self._qdrant.client
        name = self._qdrant.collection

        points = []
        for item in batch:
            point_id = item.id or str(uuid.uuid4())
            uploaded_ids.append(point_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=item.embedding,
                    payload=item.payload.model_dump(mode="json"),
                )
            )

        for attempt in range(self._settings.bulk_max_retries):
            try:
                await client.upsert(collection_name=name, points=points, wait=False)
                return len(batch), uploaded_ids, errors
            except Exception as exc:
                delay = self._settings.bulk_retry_base_delay * (2**attempt)
                msg = f"Batch {batch_idx} attempt {attempt + 1} failed: {exc}"
                logger.warning(msg)
                if attempt == self._settings.bulk_max_retries - 1:
                    errors.append(msg)
                    return 0, [], errors
                await asyncio.sleep(delay)

        return 0, [], errors

    async def _rollback_ids(self, point_ids: list[str]) -> None:
        """Delete partially uploaded points on catastrophic failure."""
        if not point_ids:
            return
        try:
            await self._qdrant.client.delete(
                collection_name=self._qdrant.collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info("Rolled back %d points", len(point_ids))
        except Exception:
            logger.exception("Rollback failed for %d points", len(point_ids))

    async def load(
        self,
        items: list[BulkLoadItem],
        *,
        batch_size: int | None = None,
        parallel_workers: int | None = None,
        disable_indexing: bool = True,
        rollback_on_failure: bool = False,
    ) -> BulkLoadStats:
        """
        Bulk upsert vectors with parallel batch processing.

        Returns progress statistics including succeeded/failed counts.
        """
        start = time.perf_counter()
        dim = len(items[0].embedding) if items else self._settings.embedding_dimension
        bs = self._batch_size(batch_size, dim)
        workers = parallel_workers or self._settings.bulk_parallel_workers

        if disable_indexing:
            await self._collection.disable_indexing_for_bulk()

        batches = [items[i : i + bs] for i in range(0, len(items), bs)]
        semaphore = asyncio.Semaphore(workers)

        succeeded = 0
        failed = 0
        all_errors: list[str] = []
        all_uploaded: list[str] = []

        async def process_batch(idx: int, batch: list[BulkLoadItem]) -> tuple[int, int, list[str], list[str]]:
            async with semaphore:
                ok, uploaded, errs = await self._upsert_batch_with_retry(batch, idx)
                failed_count = len(batch) - ok
                return ok, failed_count, uploaded, errs

        results = await asyncio.gather(
            *[process_batch(i, b) for i, b in enumerate(batches)],
            return_exceptions=True,
        )

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                actual_batch_len = len(batches[idx])
                failed += actual_batch_len
                all_errors.append(f"Batch {idx} raised unhandled exception: {result}")
                bulk_metrics.record_error()
                continue
            ok, fail, uploaded, errs = result
            succeeded += ok
            failed += fail
            all_uploaded.extend(uploaded)
            all_errors.extend(errs)

        if disable_indexing and succeeded > 0:
            await self._collection.finalize_indexes()

        if rollback_on_failure and failed > 0 and succeeded == 0:
            await self._rollback_ids(all_uploaded)

        duration_ms = (time.perf_counter() - start) * 1000
        bulk_metrics.record_request(duration_ms)

        stats = BulkLoadStats(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            batches=len(batches),
            duration_ms=round(duration_ms, 2),
            errors=all_errors[:20],
        )
        logger.info(
            "Bulk load complete: %d/%d succeeded in %.0fms",
            succeeded,
            len(items),
            duration_ms,
        )
        return stats
