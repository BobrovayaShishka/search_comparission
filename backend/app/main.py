import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from app.config import get_settings, setup_logging
from app.db import Database
from app.llm import LLMClient, TokenUsage
from app.vector_store import VectorStore

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
db = Database(settings)
llm = LLMClient(settings)
store = VectorStore(settings, db, llm)

_ASK_PROMPT = """Ты — консультант интернет-магазина. Пользователь спросил: "{query}"

Найденные товары (отсортированы по цене — от дешёвых к дорогим):
{items}

Кратко (2-4 предложения) ответь на русском: какие товары подходят и почему.
Опирайся ТОЛЬКО на список выше. Если спрашивают про «недорогие» — укажи самый дешёвый из списка.
Не придумывай товары и характеристики. Без markdown."""


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting search-comparison API")
    await db.connect()
    await llm.connect()
    await store.connect()
    seed = await store.seed_if_empty()
    logger.info("Catalog: %s", seed)
    yield
    await store.close()
    await llm.close()
    await db.close()


app = FastAPI(
    title="Search Comparison API",
    version="4.0.0",
    description="Postgres FTS vs Qdrant vector + RAG. Модели: bge-m3 + qwen2.5:3b-instruct",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "postgres": await db.health_check(),
        "qdrant": await store.health_check(),
        "ollama": await llm.health_check(),
        "mock_mode": settings.mock_mode,
        "embed_model": settings.embed_model,
        "llm_model": settings.ollama_model,
    }


@app.get("/search/fulltext")
async def search_fulltext(
    q: str = Query(min_length=1, max_length=1024),
    limit: int = Query(default=10, ge=1, le=20),
) -> dict:
    return await db.search_fulltext(q, limit)


@app.get("/search/vector")
async def search_vector(
    q: str = Query(min_length=1, max_length=1024),
    limit: int = Query(default=10, ge=1, le=20),
) -> dict:
    return await store.search(q, limit)


@app.get("/compare")
async def compare(
    q: str = Query(min_length=1, max_length=1024),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    ft = await db.search_fulltext(q, limit)
    vec = await store.search(q, limit)
    ft_ms = ft["latency_ms"]
    vec_ms = vec["latency_ms"]
    winner = "fulltext" if ft_ms <= vec_ms else "vector"
    return {
        "query": q,
        "fulltext": ft,
        "vector": vec,
        "speed": {
            "winner": winner,
            "fulltext_ms": ft_ms,
            "vector_ms": vec_ms,
            "delta_ms": round(abs(ft_ms - vec_ms), 2),
            "vector_embed_ms": vec.get("latency_embed_ms"),
            "vector_qdrant_ms": vec.get("latency_qdrant_ms"),
        },
        "tokens": {
            "fulltext": None,
            "vector_embed": vec.get("tokens"),
            "note": "LLM-токены появляются только в /ask",
        },
    }


@app.get("/ask")
async def ask(
    q: str = Query(min_length=1, max_length=1024),
    limit: int = Query(default=5, ge=1, le=10),
) -> dict:
    vec = await store.search(q, limit)
    hits = sorted(
        vec["hits"],
        key=lambda h: (h.get("price") is None, h.get("price") or 0),
    )

    if not hits:
        return {
            "query": q,
            "answer": "По вашему запросу подходящих товаров не найдено.",
            "sources": [],
            "latency_ms": vec["latency_ms"],
            "tokens": vec.get("tokens"),
        }

    items = []
    for i, hit in enumerate(hits, start=1):
        price = f", цена {hit['price']:.0f} ₽" if hit.get("price") else ""
        desc = f" — {hit['description']}" if hit.get("description") else ""
        items.append(f"{i}. {hit['name']} ({hit.get('category', '')}{price}){desc}")

    prompt = _ASK_PROMPT.format(query=q, items="\n".join(items))
    gen = await llm.generate(prompt)

    total_tokens = TokenUsage(
        embed_tokens=vec.get("tokens", {}).get("embed_tokens", 0),
        prompt_tokens=gen.tokens.prompt_tokens,
        completion_tokens=gen.tokens.completion_tokens,
    )

    return {
        "query": q,
        "answer": gen.text,
        "sources": hits,
        "latency_ms": round(vec["latency_ms"] + gen.latency_ms, 2),
        "latency_embed_ms": vec.get("latency_embed_ms"),
        "latency_qdrant_ms": vec.get("latency_qdrant_ms"),
        "latency_llm_ms": gen.latency_ms,
        "tokens": total_tokens.to_dict(),
        "tokens_breakdown": {
            "embed": vec.get("tokens"),
            "llm_prompt": gen.tokens.prompt_tokens,
            "llm_completion": gen.tokens.completion_tokens,
        },
    }
