import logging

from fastapi import APIRouter, HTTPException

from app.dependencies import collection_service, product_service, search_service
from app.models.schemas import (
    BulkLoadRequest,
    BulkLoadStats,
    ProductCreate,
    ProductUpdate,
    SearchFilters,
    SearchResponse,
    TextSearchRequest,
    VectorSearchRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["products"])


@router.post("", status_code=201)
async def create_product(payload: ProductCreate) -> dict:
    """Create product — content stored in Qdrant payload only."""
    return await product_service.create(payload)


@router.get("/browse")
async def browse_products(
    tenant_id: str | None = None,
    category: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Offset-based pagination via Qdrant scroll (non-vector listing)."""
    filters = SearchFilters(tenant_id=tenant_id, category=category)
    return await search_service.scroll(filters, limit=limit, offset=offset)


@router.post("/search", response_model=SearchResponse)
async def search_by_text(payload: TextSearchRequest) -> SearchResponse:
    """Semantic search: text → embedding → Qdrant (primary user-facing endpoint)."""
    return await product_service.search_by_text(payload)


@router.post("/search/vector", response_model=SearchResponse)
async def search_by_vector(payload: VectorSearchRequest) -> SearchResponse:
    """Direct vector search — for pre-computed embeddings."""
    return await search_service.search(payload)


@router.post("/bulk", response_model=BulkLoadStats)
async def bulk_load(payload: BulkLoadRequest) -> BulkLoadStats:
    """Mass vector ingestion with parallel batches."""
    return await product_service.bulk_load(payload)


@router.get("/{product_id}")
async def get_product(product_id: str) -> dict:
    """Get product by Qdrant point ID — no PostgreSQL content fetch."""
    product = await product_service.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.patch("/{product_id}")
async def update_product(product_id: str, payload: ProductUpdate) -> dict:
    """Update product with version increment."""
    product = await product_service.update(product_id, payload)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.delete("/{product_id}", status_code=204)
async def delete_product(product_id: str, hard: bool = False) -> None:
    """Soft delete (default) or hard delete."""
    if hard:
        ok = await product_service.hard_delete(product_id)
    else:
        ok = await product_service.soft_delete(product_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Product not found")
