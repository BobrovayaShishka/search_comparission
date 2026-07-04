import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """Async HTTP client for the self-hosted embeddings service."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.embeddings_url.rstrip("/"),
            timeout=self._settings.network_timeout_seconds,
        )
        logger.info("Embeddings client ready (%s)", self._settings.embeddings_url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for given texts."""
        if self._client is None:
            raise RuntimeError("Embeddings client is not initialized")

        response = await self._client.post("/embed", json={"texts": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    async def health_check(self) -> dict:
        if self._client is None:
            raise RuntimeError("Embeddings client is not initialized")
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()


def build_product_text(name: str, description: str, category: str) -> str:
    """Compose searchable text from product fields."""
    parts = [name.strip()]
    if category.strip():
        parts.append(f"Категория: {category.strip()}")
    if description.strip():
        parts.append(description.strip())
    return ". ".join(parts)
