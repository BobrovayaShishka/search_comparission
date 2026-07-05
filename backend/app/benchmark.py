"""
Бенчмарк Qdrant: recall vs latency при разных HNSW/quantization параметрах.

Запуск:
    python -m app.scripts.benchmark --vectors 50000 --queries 200

Что делает:
1. Генерирует случайные векторы (нормализованные, косинусная метрика).
2. Для каждой конфигурации (m, ef_construct, hnsw_ef, quantization) —
   создаёт отдельную коллекцию, грузит векторы, замеряет latency поиска
   и recall@k относительно brute-force (exact=True) поиска в Qdrant.
3. Печатает сводную таблицу и сохраняет JSON-отчёт.

Не использует сервисные классы приложения напрямую (чтобы не тащить
Postgres/Redis/embeddings), но переиспользует ту же схему параметров,
что и CollectionService, для согласованности с продакшн-конфигом.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, asdict

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark")

DIM = 384
K = 10

# Комбинации HNSW-параметров (m, ef_construct) для сравнения
HNSW_GRID: list[tuple[int, int]] = [
    (16, 100),
    (32, 200),   # текущий продакшн-дефолт
    (48, 300),
]

# ef_search-значения для сравнения скорость/точность на одном и том же индексе
EF_SEARCH_GRID: list[int] = [32, 64, 128, 256]

# "Сила" квантования: quantile определяет агрессивность обрезки выбросов.
# Меньший quantile = более грубое квантование = быстрее, но ниже recall.
# Значения примерно соответствуют компрессии в 2x/4x/8x/16x/32x относительно float32,
# что для INT8 scalar quantization константно (~4x), поэтому здесь варьируем
# quantile + rescore, а также добавляем "без квантования" как baseline.
QUANTIZATION_GRID: list[dict] = [
    {"label": "none", "quantile": None, "rescore": None},
    {"label": "q0.99_rescore", "quantile": 0.99, "rescore": True},
    {"label": "q0.95_rescore", "quantile": 0.95, "rescore": True},
    {"label": "q0.90_rescore", "quantile": 0.90, "rescore": True},
    {"label": "q0.99_norescore", "quantile": 0.99, "rescore": False},
]


@dataclass
class BenchResult:
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_ef_search: int
    quantization: str
    recall_at_10: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float


def _gen_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _quant_config(cfg: dict) -> models.ScalarQuantization | None:
    if cfg["quantile"] is None:
        return None
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=cfg["quantile"],
            always_ram=True,
        )
    )


async def _build_collection(
    client: AsyncQdrantClient,
    name: str,
    vectors: np.ndarray,
    m: int,
    ef_construct: int,
    quant_cfg: dict,
) -> None:
    if name in [c.name for c in (await client.get_collections()).collections]:
        await client.delete_collection(name)

    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        quantization_config=_quant_config(quant_cfg),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    batch = 500
    for i in range(0, len(vectors), batch):
        chunk = vectors[i : i + batch]
        await client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=str(uuid.uuid4()), vector=chunk[j].tolist())
                for j in range(len(chunk))
            ],
            wait=False,
        )

    # включаем индексацию и ждём готовности
    await client.update_collection(
        collection_name=name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )
    while True:
        info = await client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            break
        await asyncio.sleep(0.5)


async def _ground_truth(
    client: AsyncQdrantClient, name: str, queries: np.ndarray, k: int
) -> list[list[str]]:
    """Exact (brute-force) поиск для расчёта recall."""
    out = []
    for q in queries:
        res = await client.query_points(
            collection_name=name,
            query=q.tolist(),
            limit=k,
            search_params=models.SearchParams(exact=True),
        )
        out.append([str(p.id) for p in res.points])
    return out


async def _timed_search(
    client: AsyncQdrantClient,
    name: str,
    queries: np.ndarray,
    k: int,
    hnsw_ef: int,
    rescore: bool | None,
) -> tuple[list[list[str]], list[float]]:
    ids: list[list[str]] = []
    latencies: list[float] = []
    search_params = models.SearchParams(
        hnsw_ef=hnsw_ef,
        quantization=models.QuantizationSearchParams(rescore=rescore) if rescore is not None else None,
    )
    for q in queries:
        t0 = time.perf_counter()
        res = await client.query_points(
            collection_name=name,
            query=q.tolist(),
            limit=k,
            search_params=search_params,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        ids.append([str(p.id) for p in res.points])
    return ids, latencies


def _recall(ground_truth: list[list[str]], predicted: list[list[str]]) -> float:
    scores = []
    for gt, pred in zip(ground_truth, predicted):
        if not gt:
            continue
        scores.append(len(set(gt) & set(pred)) / len(gt))
    return round(statistics.mean(scores), 4) if scores else 0.0


def _percentiles(latencies: list[float]) -> tuple[float, float, float]:
    s = sorted(latencies)
    n = len(s)

    def p(pct: float) -> float:
        return round(s[min(int(n * pct / 100), n - 1)], 3)

    return p(50), p(95), p(99)


async def run_benchmark(
    host: str,
    grpc_port: int,
    http_port: int,
    n_vectors: int,
    n_queries: int,
) -> list[BenchResult]:
    client = AsyncQdrantClient(host=host, port=http_port, grpc_port=grpc_port, prefer_grpc=True)
    vectors = _gen_vectors(n_vectors, DIM)
    queries = _gen_vectors(n_queries, DIM, seed=123)

    results: list[BenchResult] = []

    for (m, ef_construct), quant_cfg in itertools.product(HNSW_GRID, QUANTIZATION_GRID):
        coll_name = f"bench_{m}_{ef_construct}_{quant_cfg['label']}"
        logger.info("Building collection m=%d ef_construct=%d quant=%s (%d vectors)",
                    m, ef_construct, quant_cfg["label"], n_vectors)
        await _build_collection(client, coll_name, vectors, m, ef_construct, quant_cfg)

        logger.info("Computing ground truth (exact search) for %d queries", n_queries)
        gt = await _ground_truth(client, coll_name, queries, K)

        for ef_search in EF_SEARCH_GRID:
            rescore = quant_cfg["rescore"]
            predicted, latencies = await _timed_search(client, coll_name, queries, K, ef_search, rescore)
            recall = _recall(gt, predicted)
            p50, p95, p99 = _percentiles(latencies)
            qps = round(1000 / statistics.mean(latencies), 2) if latencies else 0.0

            result = BenchResult(
                hnsw_m=m,
                hnsw_ef_construct=ef_construct,
                hnsw_ef_search=ef_search,
                quantization=quant_cfg["label"],
                recall_at_10=recall,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                qps=qps,
            )
            results.append(result)
            logger.info(
                "  ef_search=%-4d recall@10=%.4f p50=%.2fms p95=%.2fms qps=%.1f",
                ef_search, recall, p50, p95, qps,
            )

        await client.delete_collection(coll_name)

    await client.close()
    return results


def print_report(results: list[BenchResult]) -> None:
    header = f"{'m':>4} {'ef_c':>6} {'ef_s':>6} {'quant':>16} {'recall@10':>10} {'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'qps':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (-x.recall_at_10, x.p95_ms)):
        print(
            f"{r.hnsw_m:>4} {r.hnsw_ef_construct:>6} {r.hnsw_ef_search:>6} {r.quantization:>16} "
            f"{r.recall_at_10:>10.4f} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} {r.p99_ms:>8.2f} {r.qps:>8.1f}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant recall/speed benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--http-port", type=int, default=6333)
    parser.add_argument("--vectors", type=int, default=20_000, help="Число векторов в тестовой коллекции")
    parser.add_argument("--queries", type=int, default=200, help="Число тестовых запросов")
    parser.add_argument("--output", default="benchmark_report.json")
    args = parser.parse_args()

    results = await run_benchmark(
        args.host, args.grpc_port, args.http_port, args.vectors, args.queries
    )
    print_report(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    logger.info("Отчёт сохранён в %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
