import logging
import uuid
from datetime import datetime, timezone

from qdrant_client.http import models

from app.clients.embeddings_client import EmbeddingsClient, build_product_text
from app.clients.postgres_client import PostgresManager
from app.clients.qdrant_client import QdrantManager
from app.config import Settings
from app.exceptions import ConcurrentUpdateError, ProductRefWriteError
from app.models.schemas import (
    BulkLoadRequest,
    BulkLoadStats,
    ProductCreate,
    ProductPayload,
    ProductUpdate,
    TextSearchRequest,
    VectorSearchRequest,
)
from app.services.answer_service import AnswerService
from app.services.bulk_loader import BulkLoader
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

_MAX_CAS_RETRIES = 3


class ProductService:
    """
    Product lifecycle: upsert/delete in Qdrant (content) + ref in PostgreSQL (linkage).

    Never reads product content from PostgreSQL.

    Consistency notes (see also PostgresManager and ReconciliationService):
    - create()/hard_delete() perform a dual-write across two systems with no
      shared transaction. We use a pending→active handshake with compensation
      on the synchronous failure path, and leave a reconciliation job to
      handle the rarer crash-in-between case.
    - update()/soft_delete() use PostgreSQL as a compare-and-swap "lock" on
      the product's version, since Qdrant has no native conditional write.
    """

    def __init__(
        self,
        qdrant: QdrantManager,
        postgres: PostgresManager,
        embeddings: EmbeddingsClient,
        search: SearchService,
        bulk_loader: BulkLoader,
        answer: AnswerService,
        settings: Settings,
    ) -> None:
        self._qdrant = qdrant
        self._postgres = postgres
        self._embeddings = embeddings
        self._search = search
        self._bulk_loader = bulk_loader
        self._answer = answer
        self._settings = settings

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _resolve_tenant(self, tenant_id: str | None) -> str:
        return tenant_id or self._settings.demo_tenant_id

    async def _upsert_point(self, point_id: str, embedding: list[float], payload: ProductPayload) -> None:
        await self._qdrant.client.upsert(
            collection_name=self._qdrant.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload.model_dump(mode="json"),
                )
            ],
        )

    # Фиксированный namespace для UUID5 — детерминированно превращает
    # (tenant_id, idempotency_key) в один и тот же point_id при повторных
    # вызовах (например, retry клиента на таймауте). Без этого каждый retry
    # генерировал новый uuid4() и создавал дубликат товара.
    _IDEMPOTENCY_NAMESPACE = uuid.UUID("6f6a1e8e-2b1e-4c7a-9f0a-6d2b6a9d9b1a")

    def _derive_point_id(self, tenant_id: str, idempotency_key: str | None) -> str:
        if idempotency_key is None:
            return str(uuid.uuid4())
        return str(uuid.uuid5(self._IDEMPOTENCY_NAMESPACE, f"{tenant_id}:{idempotency_key}"))

    async def create(self, payload: ProductCreate) -> dict:
        """
        Create product: Postgres pending ref → Qdrant upsert (source of truth
        for content) → Postgres confirm active.

        Idempotent when `idempotency_key` is provided: the point_id is derived
        deterministically from (tenant_id, idempotency_key), so a client retry
        after a timeout reuses the same ID instead of minting a new uuid4()
        and creating a duplicate product. If that point already exists and is
        active, we short-circuit and return it as-is without re-embedding or
        re-writing — a true idempotent replay, not just "won't duplicate".

        If the Qdrant write fails, the pending ref is deleted (compensation)
        so we never leave a ref pointing at content that doesn't exist.
        If the final "confirm active" step fails after retries, the ref stays
        'pending' and ReconciliationService resolves it later — the point
        itself is already correctly written, so this is a bookkeeping gap,
        not a data-loss risk.
        """
        tenant_id = self._resolve_tenant(payload.tenant_id)
        point_id = self._derive_point_id(tenant_id, payload.idempotency_key)
        now = self._now()

        if payload.idempotency_key is not None:
            existing = await self.get(point_id)
            if existing is not None:
                logger.info("Idempotent replay for key %s — returning existing product %s", payload.idempotency_key, point_id)
                return existing

        # Step 1: cheap, transactional, safe to abort on failure — nothing
        # written to Qdrant yet, so no compensation needed if this fails.
        await self._postgres.insert_pending_ref(tenant_id, point_id, payload.sku)

        text = build_product_text(payload.name, payload.description, payload.category)
        embedding = (await self._embeddings.embed([text]))[0]

        product_payload = ProductPayload(
            tenant_id=tenant_id,
            name=payload.name,
            description=payload.description,
            category=payload.category,
            price=payload.price,
            sku=payload.sku,
            metadata=payload.metadata,
            is_active=True,
            is_deleted=False,
            version=1,
            created_at=now,
            updated_at=now,
        )

        # Step 2: the actual content write. Qdrant is the source of truth.
        try:
            await self._upsert_point(point_id, embedding, product_payload)
        except Exception:
            logger.warning("Qdrant write failed for %s — compensating (deleting pending ref)", point_id)
            try:
                await self._postgres.delete_product_ref(point_id)
            except Exception:
                logger.exception(
                    "Compensation failed for %s — orphan pending ref left behind, "
                    "reconciliation will find no matching Qdrant point and clean it up",
                    point_id,
                )
            raise

        # Step 3: confirm. If this fails, content is safely in Qdrant already;
        # we surface a clear error to the caller but don't roll back Qdrant —
        # rolling back a successful content write to satisfy a bookkeeping
        # write would be the wrong trade-off. Reconciliation fixes the ref later.
        try:
            await self._postgres.mark_ref_active(point_id)
        except Exception as exc:
            logger.error(
                "Failed to confirm ref active for %s after Qdrant write succeeded — "
                "left as 'pending', reconciliation will resolve it",
                point_id,
            )
            raise ProductRefWriteError(
                f"Product {point_id} was created but ref confirmation failed; "
                "it will be reconciled automatically, retry the read shortly"
            ) from exc

        return {"id": point_id, "payload": product_payload}

    async def get(self, point_id: str) -> dict | None:
        """Retrieve product by ID — Qdrant only."""
        records = await self._qdrant.client.retrieve(
            collection_name=self._qdrant.collection,
            ids=[point_id],
            with_payload=True,
        )
        if not records or not records[0].payload:
            return None
        payload = records[0].payload
        if payload.get("is_deleted"):
            return None
        return {"id": point_id, "payload": payload}

    async def update(self, point_id: str, updates: ProductUpdate) -> dict | None:
        """
        Upsert with version increment, guarded by optimistic concurrency.

        PostgreSQL's product_refs.version acts as the compare-and-swap lock:
        Qdrant has no "write only if current payload matches" primitive, so
        we use an atomic `UPDATE ... WHERE version = expected` in Postgres to
        detect concurrent writers before touching Qdrant. On conflict we
        re-read the latest state and retry, up to a bounded number of times.
        """
        for attempt in range(_MAX_CAS_RETRIES):
            existing = await self.get(point_id)
            if existing is None:
                return None

            current = ProductPayload(**existing["payload"])
            data = current.model_dump()
            for key, value in updates.model_dump(exclude_unset=True).items():
                if value is not None:
                    data[key] = value

            new_version = current.version + 1
            data["version"] = new_version
            data["updated_at"] = self._now()

            cas_ok = await self._postgres.cas_bump_version(point_id, current.version, new_version)
            if not cas_ok:
                logger.info(
                    "Version conflict updating %s (had v%d, attempt %d/%d) — retrying with fresh state",
                    point_id, current.version, attempt + 1, _MAX_CAS_RETRIES,
                )
                continue

            text = build_product_text(data["name"], data["description"], data["category"])
            embedding = (await self._embeddings.embed([text]))[0]
            new_payload = ProductPayload(**data)

            try:
                await self._upsert_point(point_id, embedding, new_payload)
            except Exception:
                # Content write failed after we already won the CAS — revert
                # the version bump so it doesn't look like content moved on
                # when it didn't, and re-raise so the caller knows to retry.
                await self._postgres.revert_version(point_id, current.version)
                raise

            if updates.sku is not None:
                await self._postgres.upsert_product_ref(current.tenant_id, point_id, updates.sku)

            return {"id": point_id, "payload": new_payload}

        raise ConcurrentUpdateError(
            f"Product {point_id} was modified concurrently {_MAX_CAS_RETRIES} times in a row; "
            "please re-fetch and retry"
        )

    async def soft_delete(self, point_id: str) -> bool:
        """
        Mark is_deleted=True in Qdrant payload (point retained for audit).

        Same CAS pattern as update() — soft-delete is a payload write too,
        and can race with a concurrent update() on the same point.
        Idempotent: re-deleting an already-deleted point still returns True.
        """
        for attempt in range(_MAX_CAS_RETRIES):
            record = await self._qdrant.client.retrieve(
                collection_name=self._qdrant.collection,
                ids=[point_id],
                with_payload=True,
            )
            if not record or not record[0].payload:
                return False

            data = record[0].payload
            current_version = data.get("version", 1)
            new_version = current_version + 1

            cas_ok = await self._postgres.cas_bump_version(point_id, current_version, new_version)
            if not cas_ok:
                logger.info(
                    "Version conflict soft-deleting %s (attempt %d/%d) — retrying",
                    point_id, attempt + 1, _MAX_CAS_RETRIES,
                )
                continue

            data["is_deleted"] = True
            data["is_active"] = False
            data["version"] = new_version
            data["updated_at"] = self._now().isoformat()

            try:
                await self._qdrant.client.set_payload(
                    collection_name=self._qdrant.collection,
                    payload=data,
                    points=[point_id],
                )
            except Exception:
                await self._postgres.revert_version(point_id, current_version)
                raise

            return True

        raise ConcurrentUpdateError(
            f"Product {point_id} was modified concurrently {_MAX_CAS_RETRIES} times in a row; "
            "please retry the delete"
        )

    async def hard_delete(self, point_id: str) -> bool:
        """
        Permanently remove point from Qdrant and ref from PostgreSQL.

        Qdrant delete happens first (it's the source of truth for content —
        once it's gone, the product no longer "exists" for any reader).
        If the Postgres ref delete then fails after retries, we leave an
        orphan ref row; reconciliation detects it (point missing from Qdrant)
        and cleans it up on the next run.
        """
        await self._qdrant.client.delete(
            collection_name=self._qdrant.collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        try:
            await self._postgres.delete_product_ref(point_id)
        except Exception:
            logger.error(
                "Qdrant point %s deleted but Postgres ref cleanup failed — "
                "orphan ref will be removed by reconciliation",
                point_id,
            )
        return True

    async def search_by_text(self, request: TextSearchRequest):
        """
        Text → embedding → vector search → (опционально) человеческий ответ через LLM.

        Если request.generate_answer=False (по умолчанию) — возвращаются только
        сырые результаты поиска, без обращения к Ollama.
        """
        embedding = (await self._embeddings.embed([request.query]))[0]
        response = await self._search.search(
            VectorSearchRequest(
                embedding=embedding,
                filters=request.filters,
                limit=request.limit,
                offset=request.offset,
                score_threshold=request.score_threshold,
                hnsw_ef=request.hnsw_ef,
            )
        )

        if request.generate_answer:
            response.answer = await self._answer.build_answer(request.query, response.hits)

        return response

    async def bulk_load(self, request: BulkLoadRequest) -> BulkLoadStats:
        """Delegate to BulkLoader for mass ingestion."""
        return await self._bulk_loader.load(
            request.items,
            batch_size=request.batch_size,
            parallel_workers=request.parallel_workers,
        )
