import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from qdrant_client.http import models

from app.clients.embeddings_client import EmbeddingsClient, build_product_text
from app.clients.postgres_client import PostgresManager
from app.clients.qdrant_client import QdrantManager
from app.config import Settings
from app.models.schemas import (
    BulkLoadItem,
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


class ProductService:
    """
    Product lifecycle: upsert/delete in Qdrant (content) + ref in PostgreSQL (linkage).

    Never reads product content from PostgreSQL.
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

    async def create(self, payload: ProductCreate) -> dict:
        """Create product: embed → Qdrant upsert → PostgreSQL ref."""
        tenant_id = self._resolve_tenant(payload.tenant_id)
        point_id = str(uuid.uuid4())
        now = self._now()

        text = build_product_text(payload.name, payload.description, payload.category)
        embedding = (await self._embeddings.embed([text]))[0]

        product_payload = ProductPayload(
            tenant_id=tenant_id,
            name=payload.name,
            description=payload.description,
            category=payload.category,
            price=float(payload.price) if payload.price is not None else None,
            sku=payload.sku,
            metadata=payload.metadata,
            is_active=True,
            is_deleted=False,
            version=1,
            created_at=now,
            updated_at=now,
        )

        await self._qdrant.client.upsert(
            collection_name=self._qdrant.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=product_payload.model_dump(mode="json"),
                )
            ],
        )

        await self._postgres.upsert_product_ref(tenant_id, point_id, payload.sku)

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
        """Upsert with version increment."""
        existing = await self.get(point_id)
        if existing is None:
            return None

        current = ProductPayload(**existing["payload"])
        data = current.model_dump()
        for key, value in updates.model_dump(exclude_unset=True).items():
            if key == "price" and value is not None:
                data[key] = float(value)
            elif value is not None:
                data[key] = value

        data["version"] = current.version + 1
        data["updated_at"] = self._now()

        text = build_product_text(data["name"], data["description"], data["category"])
        embedding = (await self._embeddings.embed([text]))[0]
        new_payload = ProductPayload(**data)

        await self._qdrant.client.upsert(
            collection_name=self._qdrant.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=new_payload.model_dump(mode="json"),
                )
            ],
        )

        if updates.sku is not None:
            await self._postgres.upsert_product_ref(current.tenant_id, point_id, updates.sku)

        return {"id": point_id, "payload": new_payload}

    async def soft_delete(self, point_id: str) -> bool:
        """
        Mark is_deleted=True in Qdrant payload (point retained for audit).

        Idempotent: re-deleting an already-deleted point still returns True
        (version is bumped again) rather than reporting "not found", since
        the point genuinely exists — only self.get() hides it once deleted.
        """
        record = await self._qdrant.client.retrieve(
            collection_name=self._qdrant.collection,
            ids=[point_id],
            with_payload=True,
        )
        if not record or not record[0].payload:
            return False

        data = record[0].payload
        data["is_deleted"] = True
        data["is_active"] = False
        data["version"] = data.get("version", 1) + 1
        data["updated_at"] = self._now().isoformat()

        await self._qdrant.client.set_payload(
            collection_name=self._qdrant.collection,
            payload=data,
            points=[point_id],
        )
        return True

    async def hard_delete(self, point_id: str) -> bool:
        """Permanently remove point from Qdrant and ref from PostgreSQL."""
        await self._qdrant.client.delete(
            collection_name=self._qdrant.collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        await self._postgres.delete_product_ref(point_id)
        return True

    async def search_by_text(self, request: TextSearchRequest):
        """
        Text → embedding → vector search → (опционально) человеческий ответ через LLM.

        Если request.generate_answer=False (по умолчанию) — возвращаются только
        сырые результаты поиска, без обращения к Ollama. Это основной,
        быстрый путь. Флаг включают точечно там, где нужен готовый ответ
        для пользователя, а не просто список товаров.
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
