import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async HTTP client for a local/self-hosted Ollama instance.

    Used to turn raw Qdrant search hits into a short human-readable answer.
    Free, runs locally — no external API costs. Recommended lightweight
    models: qwen2.5:0.5b, qwen2.5:1.5b, llama3.2:1b (fast on CPU).

    Any failure (Ollama down, timeout, model not pulled) is caught by the
    caller (AnswerService) and degraded to "no answer" — never breaks search.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        if not self._settings.ollama_enabled:
            logger.info("Ollama disabled by config")
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_url.rstrip("/"),
            timeout=self._settings.ollama_timeout_seconds,
        )
        logger.info("Ollama client ready (%s, model=%s)", self._settings.ollama_url, self._settings.ollama_model)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(self, prompt: str) -> str:
        """Single-shot generation. Raises on any failure — caller must catch."""
        if self._client is None:
            raise RuntimeError("Ollama client is not initialized")

        response = await self._client.post(
            "/api/generate",
            json={
                "model": self._settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": self._settings.ollama_max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("response") or "").strip()

    async def health_check(self) -> dict:
        if not self._settings.ollama_enabled:
            return {"status": "disabled"}
        if self._client is None:
            return {"status": "error", "detail": "not initialized"}
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            models = [m.get("name") for m in response.json().get("models", [])]
            model_pulled = self._settings.ollama_model in models
            return {
                "status": "ok" if model_pulled else "degraded",
                "model": self._settings.ollama_model,
                "model_available": model_pulled,
                "available_models": models,
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
