from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ProductPayload(BaseModel):
    """Full display content stored in Qdrant payload — never fetched from PostgreSQL."""

    tenant_id: str
    name: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=4096)
    category: str = Field(default="", max_length=256)
    price: float | None = Field(default=None, ge=0)
    sku: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    is_deleted: bool = False
    version: int = 1
    created_at: datetime
    updated_at: datetime


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=4096)
    category: str = Field(default="", max_length=256)
    price: Decimal | None = Field(default=None, ge=0)
    sku: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Клиентский ключ идемпотентности (например, UUID, сгенерированный один раз "
            "на попытку создания). Повторный вызов create() с тем же ключом и тем же "
            "tenant_id — например, из-за retry на таймауте — вернёт тот же продукт, "
            "а не создаст дубликат."
        ),
    )


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=4096)
    category: str | None = Field(default=None, max_length=256)
    price: Decimal | None = Field(default=None, ge=0)
    sku: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None


class SearchFilters(BaseModel):
    tenant_id: str | None = None
    category: str | None = None
    is_active: bool | None = True
    is_deleted: bool | None = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def empty_metadata(cls, v: Any) -> dict[str, Any]:
        return v or {}


class VectorSearchRequest(BaseModel):
    """Primary search endpoint — accepts pre-computed embedding."""

    embedding: list[float] = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hnsw_ef: int | None = Field(default=None, ge=16, le=512)


class TextSearchRequest(BaseModel):
    """Convenience endpoint — text query converted to embedding."""

    query: str = Field(min_length=1, max_length=1024)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hnsw_ef: int | None = Field(default=None, ge=16, le=512)
    generate_answer: bool = Field(
        default=False,
        description="True — сгенерировать человеческий ответ поверх результатов через Ollama. "
        "False — вернуть только сырые результаты поиска, без описания моделью.",
    )


class SearchHit(BaseModel):
    id: str
    score: float
    payload: ProductPayload


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    total: int
    limit: int
    offset: int
    latency_ms: float
    cache_hit: bool = False
    answer: str | None = Field(
        default=None,
        description="Человекочитаемый ответ от LLM (Ollama), если был запрошен generate_answer=True.",
    )


class BulkLoadItem(BaseModel):
    id: str | None = None
    embedding: list[float]
    payload: ProductPayload


class BulkLoadRequest(BaseModel):
    items: list[BulkLoadItem] = Field(min_length=1)
    batch_size: int | None = Field(default=None, ge=100, le=500)
    parallel_workers: int | None = Field(default=None, ge=1, le=16)


class BulkLoadStats(BaseModel):
    total: int
    succeeded: int
    failed: int
    batches: int
    duration_ms: float
    errors: list[str] = Field(default_factory=list)


class CollectionMigrateRequest(BaseModel):
    recreate: bool = False
    enable_quantization: bool = True


class HealthResponse(BaseModel):
    status: str
    postgres: dict[str, Any]
    qdrant: dict[str, Any]
    redis: dict[str, Any]
    embeddings: dict[str, Any]
    metrics: dict[str, Any]
    payload_indexes: list[dict[str, Any]]
    ollama: dict[str, Any] = Field(default_factory=lambda: {"status": "disabled"})
