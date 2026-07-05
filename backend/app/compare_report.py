"""
Сравнительный отчёт: релевантность + скорость + токены.

Запуск:
    docker compose exec backend python -m app.compare_report
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEMO_QUERIES = [
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
]

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "comparison.md"
BASE_URL = "http://localhost:8000"


def _format_hits(hits: list[dict], max_items: int = 3) -> str:
    if not hits:
        return "_(пусто)_"
    lines = []
    for h in hits[:max_items]:
        price = f", {h.get('price', 0):.0f} ₽" if h.get("price") else ""
        lines.append(
            f"- **{h.get('name', '?')}** ({h.get('category', '')}{price}) — score {h.get('score', 0):.3f}"
        )
    return "\n".join(lines)


def _format_tokens(tokens: dict | None) -> str:
    if not tokens:
        return "—"
    return (
        f"embed={tokens.get('embed_tokens', 0)}, "
        f"prompt={tokens.get('prompt_tokens', 0)}, "
        f"completion={tokens.get('completion_tokens', 0)}, "
        f"total={tokens.get('total_tokens', 0)}"
    )


async def run_report(base_url: str = BASE_URL, output: Path = OUTPUT) -> Path:
    sections: list[str] = [
        "# Сравнение полнотекстового и векторного поиска",
        "",
        "Модели: **bge-m3** (эмбеддинги) + **qwen2.5:3b-instruct** (RAG `/ask`).",
        "",
        f"Базовый URL: `{base_url}`",
        "",
    ]
    speed_rows: list[str] = [
        "| # | Запрос | FTS ms | Vec ms | Быстрее | Embed tokens |",
        "|---:|---|---:|---:|---|---:|",
    ]

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        for row_num, query in enumerate(DEMO_QUERIES, start=1):
            try:
                compare = await client.get("/compare", params={"q": query, "limit": 5})
                compare.raise_for_status()
                data = compare.json()
            except Exception as exc:
                sections.extend([f"\n## «{query}»\n", f"_Ошибка: {exc}_\n"])
                continue

            ft = data.get("fulltext", {})
            vec = data.get("vector", {})
            speed = data.get("speed", {})
            embed_tokens = (vec.get("tokens") or {}).get("embed_tokens", 0)

            speed_rows.append(
                f"| {row_num} | {query} | {ft.get('latency_ms', 0)} | {vec.get('latency_ms', 0)} | "
                f"{speed.get('winner', '?')} | {embed_tokens} |"
            )

            sections.extend([
                f"\n## «{query}»",
                "",
                f"**Полнотекст** (режим `{ft.get('mode', '?')}`, **{ft.get('latency_ms', 0)} ms**):",
                _format_hits(ft.get("hits", [])),
                "",
                f"**Векторный** (**{vec.get('latency_ms', 0)} ms**: "
                f"embed {vec.get('latency_embed_ms', 0)} ms + qdrant {vec.get('latency_qdrant_ms', 0)} ms, "
                f"tokens: {_format_tokens(vec.get('tokens'))}):",
                _format_hits(vec.get("hits", [])),
                "",
                f"**Скорость:** {speed.get('winner')} быстрее на {speed.get('delta_ms', 0)} ms",
                "",
            ])

            try:
                ask = await client.get("/ask", params={"q": query, "limit": 5})
                ask.raise_for_status()
                ask_data = ask.json()
                sections.extend([
                    f"**RAG /ask** ({ask_data.get('latency_ms', 0)} ms, tokens: {_format_tokens(ask_data.get('tokens'))}):",
                    f"> {ask_data.get('answer', '')[:500]}",
                    "",
                ])
            except Exception as exc:
                sections.append(f"_RAG недоступен: {exc}_\n")

    sections[7:7] = ["## Сводка скорости", ""] + speed_rows + [""]

    sections.extend([
        "## Выводы",
        "",
        "- **Релевантность:** векторный лучше на смысловых запросах и опечатках; FTS — на точных названиях.",
        "- **Скорость:** FTS почти всегда быстрее — нет вызова embed; узкое место vector — bge-m3 через Ollama.",
        "- **Токены:** embed-токены растут с длиной запроса; /ask добавляет prompt+completion от qwen2.5:3b.",
        "- **bge-m3** даёт лучшее разделение score на русском, чем лёгкие multilingual-модели.",
        "",
        "Подробный бенчмарк скорости: `python -m app.benchmark_fts_vs_vector` → `data/benchmark_fts_vs_vector.md`.",
        "",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Report written to %s", output)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_report())


if __name__ == "__main__":
    main()
