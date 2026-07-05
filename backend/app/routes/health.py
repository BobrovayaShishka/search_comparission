import logging

from fastapi import APIRouter

from app.dependencies import (
    cache,
    collection_service,
    embeddings,
    ollama,
    postgres,
    qdrant,
    reconciliation_service,
)
from app.models.schemas import CollectionMigrateRequest, HealthResponse
from app.monitoring.metrics import bulk_metrics, search_metrics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Full system health: DB connections, indexes, metrics."""
    pg_status = await postgres.health_check()
    qd_status = await qdrant.health_check()
    redis_status = await cache.health_check()
    emb_status = await embeddings.health_check()
    ollama_status = await ollama.health_check()
    indexes = await qdrant.check_payload_indexes()

    # Ollama намеренно необязателен: "degraded"/"error" по нему не переводит
    # весь сервис в degraded — поиск работает и без LLM-ответа.
    all_ok = all(
        s.get("status") in ("ok", "disabled")
        for s in [pg_status, qd_status, redis_status, emb_status]
    )

    # Индексация "застряла" отключённой (m=0) вне активной bulk-загрузки —
    # обычно значит, что bulk_load() упал до своего finally-блока в старой
    # версии кода, или сейчас реально идёт загрузка. В обоих случаях стоит
    # явно показать это в /health, а не молчать.
    collection_info = qd_status.get("collection_info") or {}
    indexing_disabled = bool(collection_info.get("indexing_disabled"))
    if indexing_disabled:
        all_ok = False
        qd_status = {**qd_status, "warning": "HNSW indexing is disabled (m=0) on the live collection"}

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        postgres=pg_status,
        qdrant=qd_status,
        redis=redis_status,
        embeddings=emb_status,
        metrics={
            "search": search_metrics.snapshot(),
            "bulk": bulk_metrics.snapshot(),
        },
        payload_indexes=indexes,
        ollama=ollama_status,
    )


@router.post("/admin/migrate")
async def migrate_collection(payload: CollectionMigrateRequest) -> dict:
    """Create/recreate Qdrant collection with optimal production parameters."""
    return await collection_service.create_collection(
        recreate=payload.recreate,
        enable_quantization=payload.enable_quantization,
    )


@router.post("/admin/finalize-indexes")
async def finalize_indexes() -> dict:
    """Re-enable HNSW after bulk import."""
    return await collection_service.finalize_indexes()


@router.post("/admin/reconcile")
async def reconcile(older_than_seconds: int = 300) -> dict:
    """
    Resolve product_refs stuck in 'pending' after a crashed/interrupted
    dual-write (Qdrant write succeeded or failed, but PostgreSQL was never
    told which). Safe to call repeatedly; run on a schedule in production.
    """
    return await reconciliation_service.reconcile_pending_refs(older_than_seconds)
