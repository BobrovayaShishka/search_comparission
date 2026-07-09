"""
Эмбеддинги + генерация:
- Ollama (локально): bge-m3 + qwen2.5:3b-instruct
- Dockhost LLM Inference: OpenAI-совместимый API (INFERENCE_API_KEY)
- MOCK_MODE=true — hash-эмбеддинги и заглушка без сетевых вызовов
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


@dataclass
class TokenUsage:
    embed_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.embed_tokens + self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "embed_tokens": self.embed_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0


@dataclass
class GenerateResult:
    text: str
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0


def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.3))


def _mock_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dim:
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) >= dim:
                break
        digest = hashlib.sha256(digest).digest()
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def build_product_text(name: str, description: str, category: str) -> str:
    parts = [name.strip()]
    if category.strip():
        parts.append(f"Категория: {category.strip()}")
    if description.strip():
        parts.append(description.strip())
    return ". ".join(parts)


def _extract_assistant_text(message: dict | None, choice: dict | None = None) -> str:
    """OpenAI / Dockhost / Qwen: content может быть пустым, текст — в reasoning."""
    if message:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str) and part.strip():
                    chunks.append(part.strip())
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks).strip()

        for key in ("reasoning", "reasoning_content"):
            val = message.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        for item in message.get("reasoning_details") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"reasoning.text", "text", "summary_text"}:
                text = item.get("text") or item.get("summary")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    if choice:
        legacy = choice.get("text")
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()

    return ""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def uses_inference(self) -> bool:
        return bool(self._settings.inference_api_key.strip())

    @property
    def chat_model(self) -> str:
        if self.uses_inference:
            return self._settings.inference_chat_model
        return self._settings.ollama_model

    async def connect(self) -> None:
        if self._settings.mock_mode:
            logger.info("LLM client in MOCK_MODE")
            return
        if self.uses_inference:
            self._client = httpx.AsyncClient(
                base_url=self._settings.inference_base_url.rstrip("/"),
                timeout=self._settings.inference_timeout_seconds,
                headers={
                    "Authorization": f"Bearer {self._settings.inference_api_key.strip()}",
                    "Content-Type": "application/json",
                },
            )
            logger.info(
                "LLM client: Dockhost Inference (%s, chat=%s)",
                self._settings.embed_model,
                self.chat_model,
            )
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_url.rstrip("/"),
            timeout=self._settings.ollama_timeout_seconds,
        )
        logger.info("LLM client: Ollama (%s)", self._settings.ollama_url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str]) -> EmbedResult:
        start = time.perf_counter()
        if self._settings.mock_mode:
            vectors = [
                _mock_embedding(t, self._settings.embedding_dimension) for t in texts
            ]
            tokens = TokenUsage(embed_tokens=sum(_estimate_tokens(t) for t in texts))
            return EmbedResult(
                vectors=vectors,
                tokens=tokens,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        if self._client is None:
            raise RuntimeError("LLM client is not initialized")

        if self.uses_inference:
            return await self._embed_inference(texts, start)
        return await self._embed_ollama(texts, start)

    async def _embed_ollama(self, texts: list[str], start: float) -> EmbedResult:
        response = await self._client.post(
            "/api/embed",
            json={"model": self._settings.embed_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        vectors = data["embeddings"]
        embed_tokens = sum(_estimate_tokens(t) for t in texts)
        return EmbedResult(
            vectors=vectors,
            tokens=TokenUsage(embed_tokens=embed_tokens),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def _embed_inference(self, texts: list[str], start: float) -> EmbedResult:
        vectors: list[list[float]] = []
        embed_tokens = 0

        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            response = await self._client.post(
                "/embeddings",
                json={"model": self._settings.embed_model, "input": batch},
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            embed_tokens += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
            ordered = sorted(data.get("data") or [], key=lambda row: row.get("index", 0))
            vectors.extend(row["embedding"] for row in ordered)

        if vectors and len(vectors[0]) != self._settings.embedding_dimension:
            logger.warning(
                "Embedding dim %d != EMBEDDING_DIMENSION=%d — обновите env и пересидите Qdrant",
                len(vectors[0]),
                self._settings.embedding_dimension,
            )

        if not embed_tokens:
            embed_tokens = sum(_estimate_tokens(t) for t in texts)

        return EmbedResult(
            vectors=vectors,
            tokens=TokenUsage(embed_tokens=embed_tokens),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def generate(self, prompt: str) -> GenerateResult:
        start = time.perf_counter()
        if self._settings.mock_mode:
            return GenerateResult(
                text=(
                    "Демо-ответ (MOCK_MODE). Включите Ollama/Inference API "
                    "и выключите MOCK_MODE для реальной генерации."
                ),
                tokens=TokenUsage(
                    prompt_tokens=_estimate_tokens(prompt),
                    completion_tokens=20,
                ),
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        if self._client is None:
            raise RuntimeError("LLM client is not initialized")

        if self.uses_inference:
            return await self._generate_inference(prompt, start)
        return await self._generate_ollama(prompt, start)

    async def _generate_ollama(self, prompt: str, start: float) -> GenerateResult:
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self._settings.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": self._settings.ollama_max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        tokens = TokenUsage(
            prompt_tokens=int(data.get("prompt_eval_count") or _estimate_tokens(prompt)),
            completion_tokens=int(data.get("eval_count") or _estimate_tokens(text)),
        )
        return GenerateResult(
            text=text,
            tokens=tokens,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def _generate_inference(self, prompt: str, start: float) -> GenerateResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты консультант магазина. Отвечай кратко на русском, "
                    "2–4 предложения. Без markdown и без блока рассуждений."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        base_payload: dict = {
            "model": self.chat_model,
            "messages": messages,
            "max_tokens": self._settings.inference_max_tokens,
            "temperature": 0.2,
        }

        payloads = [base_payload]
        if self.chat_model.startswith("qwen/"):
            with_reasoning = {**base_payload, "reasoning": {"exclude": True}}
            payloads = [with_reasoning, base_payload]

        data: dict | None = None
        last_exc: Exception | None = None
        for body in payloads:
            try:
                response = await self._client.post("/chat/completions", json=body)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in {400, 422} and body is not base_payload:
                    continue
                raise

        if data is None:
            raise last_exc or RuntimeError("Inference chat failed")

        choices = data.get("choices") or []
        text = ""
        if choices:
            text = _extract_assistant_text(choices[0].get("message"), choices[0])

        if not text:
            logger.warning(
                "Inference chat returned empty text (model=%s, finish=%s)",
                self.chat_model,
                choices[0].get("finish_reason") if choices else None,
            )

        usage = data.get("usage") or {}
        tokens = TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or _estimate_tokens(prompt)),
            completion_tokens=int(usage.get("completion_tokens") or _estimate_tokens(text)),
        )
        return GenerateResult(
            text=text,
            tokens=tokens,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def health_check(self) -> dict:
        if self._settings.mock_mode:
            return {
                "status": "mock",
                "provider": "mock",
                "embed_model": self._settings.embed_model,
                "llm_model": self.chat_model,
            }
        if self._client is None:
            return {"status": "error", "detail": "not initialized"}
        if self.uses_inference:
            return await self._health_inference()
        return await self._health_ollama()

    async def _health_ollama(self) -> dict:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]

            def _has(name: str) -> bool:
                return any(name in m for m in models)

            return {
                "status": "ok" if _has(self._settings.embed_model) and _has(self._settings.ollama_model) else "degraded",
                "provider": "ollama",
                "embed_model": self._settings.embed_model,
                "embed_available": _has(self._settings.embed_model),
                "llm_model": self._settings.ollama_model,
                "llm_available": _has(self._settings.ollama_model),
                "available_models": models,
            }
        except Exception as exc:
            return {"status": "error", "provider": "ollama", "detail": str(exc)}

    async def _health_inference(self) -> dict:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            models = [m.get("id", "") for m in response.json().get("data", [])]

            def _has(name: str) -> bool:
                return name in models

            embed_ok = _has(self._settings.embed_model)
            chat_ok = _has(self.chat_model)
            return {
                "status": "ok" if embed_ok and chat_ok else "degraded",
                "provider": "inference",
                "embed_model": self._settings.embed_model,
                "embed_available": embed_ok,
                "llm_model": self.chat_model,
                "llm_available": chat_ok,
            }
        except Exception as exc:
            return {"status": "error", "provider": "inference", "detail": str(exc)}
