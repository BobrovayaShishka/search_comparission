"""
Бенчмарк скорости: Postgres FTS vs embed+Qdrant на одних запросах.

Запуск:
    docker compose exec backend python -m app.benchmark_fts_vs_vector
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

QUERIES = [
    "что-нибудь для приготовления кофе",
    "гаджет для уборки квартиры",
    "устройство чтобы слушать музыку без проводов",
    "кросовки",
    "кроссовки для бега",
    "айфон",
    "Gel-Nimbus",
    "ASUS VivoBook",
    "чайник",
    "кофе",
    "беспроводные наушники",
    "недорогие беговые кроссовки",
]

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_fts_vs_vector.md"
BASE_URL = "http://localhost:8000"
ROUNDS = 5


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return round(s[idx], 2)


async def _time_get(client: httpx.AsyncClient, path: str, params: dict) -> tuple[float, dict]:
    start = time.perf_counter()
    response = await client.get(path, params=params)
    response.raise_for_status()
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    return elapsed, response.json()


async def run_benchmark(base_url: str = BASE_URL, output: Path = OUTPUT) -> Path:
    ft_times: list[float] = []
    vec_times: list[float] = []
    vec_embed_times: list[float] = []
    vec_qdrant_times: list[float] = []
    per_query: list[dict] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        # прогрев
        await client.get("/search/fulltext", params={"q": "кофе", "limit": 5})
        await client.get("/search/vector", params={"q": "кофе", "limit": 5})

        for query in QUERIES:
            q_ft: list[float] = []
            q_vec: list[float] = []
            q_embed: list[float] = []
            q_qdrant: list[float] = []

            for _ in range(ROUNDS):
                ms, _ = await _time_get(client, "/search/fulltext", {"q": query, "limit": 5})
                q_ft.append(ms)
                ft_times.append(ms)

                ms, data = await _time_get(client, "/search/vector", {"q": query, "limit": 5})
                q_vec.append(ms)
                vec_times.append(ms)
                q_embed.append(data.get("latency_embed_ms", 0))
                q_qdrant.append(data.get("latency_qdrant_ms", 0))
                vec_embed_times.append(data.get("latency_embed_ms", 0))
                vec_qdrant_times.append(data.get("latency_qdrant_ms", 0))

            per_query.append({
                "query": query,
                "ft_p50": _percentile(q_ft, 50),
                "vec_p50": _percentile(q_vec, 50),
                "embed_p50": _percentile(q_embed, 50),
                "qdrant_p50": _percentile(q_qdrant, 50),
                "winner": "fulltext" if statistics.mean(q_ft) <= statistics.mean(q_vec) else "vector",
            })

    ft_wins = sum(1 for r in per_query if r["winner"] == "fulltext")
    overall_winner = "fulltext" if statistics.mean(ft_times) <= statistics.mean(vec_times) else "vector"

    lines = [
        "# Бенчмарк скорости: FTS vs Vector",
        "",
        f"URL: `{base_url}`, запросов: {len(QUERIES)}, раундов на запрос: {ROUNDS}",
        "",
        "## Сводка (all queries)",
        "",
        "| Метрика | Fulltext (Postgres) | Vector (embed+Qdrant) |",
        "|---|---:|---:|",
        f"| mean ms | {statistics.mean(ft_times):.2f} | {statistics.mean(vec_times):.2f} |",
        f"| p50 ms | {_percentile(ft_times, 50):.2f} | {_percentile(vec_times, 50):.2f} |",
        f"| p95 ms | {_percentile(ft_times, 95):.2f} | {_percentile(vec_times, 95):.2f} |",
        "",
        f"**Быстрее в среднем:** `{overall_winner}` ({ft_wins}/{len(per_query)} запросов выиграл fulltext)",
        "",
        "### Декомпозиция vector latency",
        "",
        f"- embed (bge-m3): p50 **{_percentile(vec_embed_times, 50):.2f} ms**",
        f"- qdrant search: p50 **{_percentile(vec_qdrant_times, 50):.2f} ms**",
        "",
        "## По запросам (p50 ms)",
        "",
        "| Запрос | FTS p50 | Vector p50 | Embed p50 | Qdrant p50 | Быстрее |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in per_query:
        lines.append(
            f"| {row['query']} | {row['ft_p50']} | {row['vec_p50']} | "
            f"{row['embed_p50']} | {row['qdrant_p50']} | {row['winner']} |"
        )

    lines.extend([
        "",
        "## Вывод",
        "",
        "- **Fulltext** обычно быстрее на коротких лексических запросах без эмбеддинга.",
        "- **Vector** добавляет latency эмбеддинга (bge-m3 через Ollama) — основной overhead.",
        "- На ~65 товарах Qdrant-поиск сам по себе очень быстрый; узкое место — embed.",
        "",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Benchmark written to %s", output)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
