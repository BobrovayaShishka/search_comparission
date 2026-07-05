import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import Settings

logger = logging.getLogger(__name__)

# Ретраим подключение к Redis с backoff ПЕРЕД тем, как упасть в in-memory
# fallback — иначе на старте контейнера (Redis ещё не готов в docker-compose)
# сервис сразу и навсегда переходит на память, даже если Redis появится
# через секунду.
_connect_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.3, max=5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class SearchCache:
    """Two-tier cache: Redis (primary) + in-memory dict fallback with TTL."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis: aioredis.Redis | None = None
        self._local: dict[str, tuple[float, Any]] = {}

    @_connect_retry
    async def _connect_redis(self) -> aioredis.Redis:
        client = aioredis.from_url(
            self._settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=self._settings.network_timeout_seconds,
        )
        await client.ping()
        return client

    async def connect(self) -> None:
        if not self._settings.cache_enabled:
            logger.info("Search cache disabled")
            return
        try:
            self._redis = await self._connect_redis()
            logger.info("Redis cache connected")
        except Exception:
            logger.warning(
                "Redis unavailable after retries, using in-memory cache fallback", exc_info=True
            )
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _evict_expired_local(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._local.items() if exp <= now]
        for key in expired:
            del self._local[key]

    async def get(self, key: str) -> Any | None:
        if not self._settings.cache_enabled:
            return None

        if self._redis is not None:
            try:
                raw = await self._redis.get(f"search:{key}")
                if raw:
                    return json.loads(raw)
            except Exception:
                logger.debug("Redis get failed", exc_info=True)

        self._evict_expired_local()
        entry = self._local.get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        return None

    async def set(self, key: str, value: Any) -> None:
        if not self._settings.cache_enabled:
            return

        serialized = json.dumps(value, default=str)

        if self._redis is not None:
            try:
                await self._redis.setex(
                    f"search:{key}",
                    self._settings.cache_ttl_seconds,
                    serialized,
                )
                return
            except Exception:
                logger.debug("Redis set failed", exc_info=True)

        self._local[key] = (
            time.monotonic() + self._settings.cache_ttl_seconds,
            value,
        )

    async def health_check(self) -> dict[str, Any]:
        if not self._settings.cache_enabled:
            return {"status": "disabled"}
        if self._redis is not None:
            try:
                await self._redis.ping()
                return {"status": "ok", "backend": "redis"}
            except Exception as exc:
                return {"status": "degraded", "backend": "redis", "error": str(exc)}
        return {"status": "ok", "backend": "memory", "entries": len(self._local)}
