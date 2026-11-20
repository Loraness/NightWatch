import asyncio
import json
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.config import settings

NUM_ROWS = 100_000
DIM = 384
NUM_QUERIES = 20
TOP_K = 5


def _generate_normalized_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def _extract_scan_node(plan: dict) -> str:
    node_type = plan.get("Node Type", "")
    if "Scan" in node_type:
        return node_type
    for child in plan.get("Plans", []):
        found = _extract_scan_node(child)
        if found:
            return found
    return node_type


async def _measure_queries(
    conn: asyncpg.Connection,
    queries: np.ndarray,
    setup_sql: str | None = None,
) -> tuple[float, list[set[int]], str]:
    latencies_ms: list[float] = []
    results: list[set[int]] = []
    sample_plan = ""

    for idx, q in enumerate(queries):
        async with conn.transaction():
            if setup_sql:
                await conn.execute(setup_sql)

            t0 = time.perf_counter()
            rows = await conn.fetch(
                f"SELECT id FROM benchmark_vectors ORDER BY embedding <=> $1 LIMIT {TOP_K}",
                q,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            results.append({r["id"] for r in rows})

            if idx == 0:
                plan_rows = await conn.fetch(
                    f"EXPLAIN (ANALYZE, FORMAT JSON) "
                    f"SELECT id FROM benchmark_vectors ORDER BY embedding <=> $1 LIMIT {TOP_K}",
                    q,
                )
                plan_json = json.loads(plan_rows[0][0])
                sample_plan = _extract_scan_node(plan_json[0]["Plan"])

    p95 = float(np.percentile(latencies_ms, 95))
    return p95, results, sample_plan


def _compute_recall(ground_truth: list[set[int]], predicted: list[set[int]]) -> float:
    hits = sum(len(gt & pred) for gt, pred in zip(ground_truth, predicted, strict=True))
    return (hits / (len(ground_truth) * TOP_K)) * 100.0


async def run_benchmark() -> None:
    dsn = settings.async_database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    await register_vector(conn)

    try:
        print(f"Generating {NUM_ROWS} normalized {DIM}-dim vectors...")
        vectors = _generate_normalized_vectors(NUM_ROWS, DIM, seed=42)
        queries = _generate_normalized_vectors(NUM_QUERIES, DIM, seed=99)

        await conn.execute("DROP TABLE IF EXISTS benchmark_vectors;")
        await conn.execute(f"CREATE TABLE benchmark_vectors (id INTEGER PRIMARY KEY, embedding vector({DIM}));")

        t0 = time.perf_counter()
        records = ((i, vectors[i]) for i in range(NUM_ROWS))
        await conn.copy_records_to_table(
            "benchmark_vectors",
            records=records,
            columns=["id", "embedding"],
        )
        await conn.execute("ANALYZE benchmark_vectors;")
        print(f"Loaded {NUM_ROWS} vectors via COPY in {time.perf_counter() - t0:.2f}s\n")

        p95_exact, ground_truth, plan_exact = await _measure_queries(conn, queries)
        print(f"[Exact / No Index]  Plan: {plan_exact:<12} | p95: {p95_exact:6.2f} ms | Recall@5: 100.0%")

        t_build_ivf = time.perf_counter()
        await conn.execute(
            "CREATE INDEX ix_bench_ivf ON benchmark_vectors "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )
        await conn.execute("ANALYZE benchmark_vectors;")
        build_ivf_s = time.perf_counter() - t_build_ivf
        p95_ivf, res_ivf, plan_ivf = await _measure_queries(
            conn, queries, setup_sql="SET LOCAL ivfflat.probes = 10;"
        )
        recall_ivf = _compute_recall(ground_truth, res_ivf)
        print(
            f"[IVFFlat lists=100] Plan: {plan_ivf:<12} | Build: {build_ivf_s:5.2f}s | "
            f"p95: {p95_ivf:6.2f} ms | Recall@5: {recall_ivf:5.1f}%"
        )
        await conn.execute("DROP INDEX ix_bench_ivf;")

        t_build_hnsw = time.perf_counter()
        await conn.execute(
            "CREATE INDEX ix_bench_hnsw ON benchmark_vectors "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
        )
        await conn.execute("ANALYZE benchmark_vectors;")
        build_hnsw_s = time.perf_counter() - t_build_hnsw
        p95_hnsw, res_hnsw, plan_hnsw = await _measure_queries(
            conn, queries, setup_sql="SET LOCAL hnsw.ef_search = 40;"
        )
        recall_hnsw = _compute_recall(ground_truth, res_hnsw)
        print(
            f"[HNSW m=16, ef=64]  Plan: {plan_hnsw:<12} | Build: {build_hnsw_s:5.2f}s | "
            f"p95: {p95_hnsw:6.2f} ms | Recall@5: {recall_hnsw:5.1f}%"
        )
    finally:
        await conn.execute("DROP TABLE IF EXISTS benchmark_vectors;")
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
