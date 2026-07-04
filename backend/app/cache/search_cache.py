import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import Settings

logger = logging.getLogger(__name__)


class SearchCache:
    """Two-tier cache: Redis (primary) + in-memory dict fallback with TTL."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis: aioredis.Redis | None = None
        self._local: dict[str, tuple[float, Any]] = {}

    async def connect(self) -> None:
        if not self._settings.cache_enabled:
            logger.info("Search cache disabled")
            return
        try:
            self._redis = aioredis.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=self._settings.network_timeout_seconds,
            )
            await self._redis.ping()
            logger.info("Redis cache connected")
        except Exception:
            logger.warning("Redis unavailable, using in-memory cache fallback", exc_info=True)
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
