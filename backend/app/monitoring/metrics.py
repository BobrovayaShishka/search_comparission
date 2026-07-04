import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """In-memory latency/QPS/error metrics for monitoring."""

    _latencies: list[float] = field(default_factory=list)
    _errors: int = 0
    _requests: int = 0
    _cache_hits: int = 0
    _window_start: float = field(default_factory=time.monotonic)
    _max_samples: int = 10_000

    def record_request(self, latency_ms: float, cache_hit: bool = False) -> None:
        self._requests += 1
        self._latencies.append(latency_ms)
        if cache_hit:
            self._cache_hits += 1
        if len(self._latencies) > self._max_samples:
            self._latencies = self._latencies[-self._max_samples :]

    def record_error(self) -> None:
        self._errors += 1
        self._requests += 1

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self._window_start, 0.001)
        latencies = sorted(self._latencies)
        n = len(latencies)

        def percentile(p: float) -> float | None:
            if n == 0:
                return None
            idx = min(int(n * p / 100), n - 1)
            return round(latencies[idx], 2)

        return {
            "requests_total": self._requests,
            "errors_total": self._errors,
            "error_rate": round(self._errors / max(self._requests, 1), 4),
            "qps": round(self._requests / elapsed, 2),
            "cache_hit_rate": round(self._cache_hits / max(self._requests, 1), 4),
            "latency_ms": {
                "p50": percentile(50),
                "p95": percentile(95),
                "p99": percentile(99),
                "avg": round(sum(latencies) / n, 2) if n else None,
            },
        }

    def reset(self) -> None:
        self._latencies.clear()
        self._errors = 0
        self._requests = 0
        self._cache_hits = 0
        self._window_start = time.monotonic()


search_metrics = MetricsCollector()
bulk_metrics = MetricsCollector()


def hash_query(embedding: list[float], filters: dict[str, Any], limit: int, offset: int) -> str:
    """Deterministic cache key from search parameters."""
    payload = json.dumps(
        {"e": [round(v, 6) for v in embedding], "f": filters, "l": limit, "o": offset},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
