import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings, setup_logging
from app.dependencies import (
    cache,
    collection_service,
    embeddings,
    postgres,
    qdrant,
)
from app.routes.health import router as health_router
from app.routes.products import router as products_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logger.info("Starting Product Search API")

    await postgres.connect()
    await qdrant.connect()
    await embeddings.connect()
    await cache.connect()

    await collection_service.create_collection(recreate=False, enable_quantization=True)

    yield

    await cache.close()
    await embeddings.close()
    await qdrant.close()
    await postgres.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Product Search API",
    version="2.0.0",
    description="High-performance vector search: Qdrant (vectors+content) + PostgreSQL (ops)",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(products_router)
