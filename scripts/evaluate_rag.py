import asyncio
import uuid
from dataclasses import dataclass

from sqlalchemy import delete

from app.database import async_session_maker
from app.models import Document, DocumentChunk
from app.services.embeddings import get_embedding_service, get_reranker_service
from app.services.search import hybrid_search, rerank_results, vector_search


@dataclass(frozen=True)
class EvalMetrics:
    recall_at_3: float
    recall_at_5: float
    mrr: float


@dataclass(frozen=True)
class CorpusChunkSpec:
    key: str
    section_header: str
    content: str


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_key: str


CORPUS_SPECS: list[CorpusChunkSpec] = [
    CorpusChunkSpec(
        key="pg_pool_overview",
        section_header="PostgreSQL Connection Pool Overview",
        content=(
            "[Section: PostgreSQL Connection Pool Overview]\n"
            "PostgreSQL connection pool issues happen frequently across Python async services during traffic spikes. "
            "Symptoms include elevated HTTP 500 latency, stalled endpoints, and database connection warnings in dashboards."
        ),
    ),
    CorpusChunkSpec(
        key="pg_pool_metrics",
        section_header="PostgreSQL Connection Pool Monitoring",
        content=(
            "[Section: PostgreSQL Connection Pool Monitoring]\n"
            "To monitor PostgreSQL connection pool usage and QueuePool wait time, inspect pg_stat_activity "
            "and export asyncpg pool checkedout vs overflow counters to Prometheus every 15 seconds."
        ),
    ),
    CorpusChunkSpec(
        key="pg_pool_leak_fix",
        section_header="Fixing QueuePool Limit Exhaustion",
        content=(
            "[Section: Fixing QueuePool Limit Exhaustion]\n"
            "Remediation for TimeoutError QueuePool limit of size 10 overflow 20 reached: audit every manual "
            "session acquisition and ensure callers always use async with async_session_maker() so connections return to the pool."
        ),
    ),
    CorpusChunkSpec(
        key="pg_code_53300",
        section_header="PostgreSQL Slot Limit Error",
        content=(
            "[Section: PostgreSQL Slot Limit Error]\n"
            "Catalog entry SQLSTATE_53300: FATAL remaining connection slots are reserved for non-replication superuser. "
            "Reduce per-worker pool_size or deploy PgBouncer in transaction pooling mode in front of Postgres."
        ),
    ),
    CorpusChunkSpec(
        key="pg_code_40P01",
        section_header="PostgreSQL Lock Order Failure",
        content=(
            "[Section: PostgreSQL Lock Order Failure]\n"
            "Catalog entry SQLSTATE_40P01: deadlock detected between concurrent transaction workers. "
            "Sort row primary keys before acquiring FOR UPDATE locks and retry the transaction up to 3 times."
        ),
    ),
    CorpusChunkSpec(
        key="pg_code_57014",
        section_header="PostgreSQL Statement Timeout Abort",
        content=(
            "[Section: PostgreSQL Statement Timeout Abort]\n"
            "Catalog entry SQLSTATE_57014: canceling statement due to statement_timeout during heavy report scan. "
            "Add a covering composite index or lower statement batch size instead of raising global timeout."
        ),
    ),
    CorpusChunkSpec(
        key="pg_code_23505",
        section_header="PostgreSQL Duplicate Key Violation",
        content=(
            "[Section: PostgreSQL Duplicate Key Violation]\n"
            "Catalog entry SQLSTATE_23505: duplicate key value violates unique constraint during idempotent webhook replay. "
            "Switch INSERT to PostgreSQL ON CONFLICT DO NOTHING and read the existing row."
        ),
    ),
    CorpusChunkSpec(
        key="pg_code_08006",
        section_header="PostgreSQL Socket Reset Failure",
        content=(
            "[Section: PostgreSQL Socket Reset Failure]\n"
            "Catalog entry SQLSTATE_08006: connection failure when Postgres pod restarted underneath idle pool sockets. "
            "Enable pool_pre_ping=True on create_async_engine so stale TCP connections are recycled automatically."
        ),
    ),
    CorpusChunkSpec(
        key="pg_pending_rollback",
        section_header="SQLAlchemy Invalid Transaction State",
        content=(
            "[Section: SQLAlchemy Invalid Transaction State]\n"
            "Catalog entry SA_ERR_PRB01: PendingRollbackError when reusing an AsyncSession after an IntegrityError. "
            "Always call await session.rollback() inside the except block before executing fallback UPDATE statements."
        ),
    ),
    CorpusChunkSpec(
        key="pg_missing_greenlet",
        section_header="SQLAlchemy Lazy Load Greenlet Crash",
        content=(
            "[Section: SQLAlchemy Lazy Load Greenlet Crash]\n"
            "Catalog entry SA_ERR_MGL02: MissingGreenlet greenlet_spawn has not been called when accessing ORM fields "
            "after await session.commit(). Configure async_sessionmaker with expire_on_commit=False."
        ),
    ),
    CorpusChunkSpec(
        key="pg_hnsw_shm",
        section_header="PostgreSQL Parallel Index Shared Memory",
        content=(
            "[Section: PostgreSQL Parallel Index Shared Memory]\n"
            "Catalog entry PGV_ERR_SHM64: DiskFullError could not resize shared memory segment /PostgreSQL to 533MB. "
            "Docker limits /dev/shm to 64MB by default; set shm_size: 1gb on the db service in docker-compose.yml."
        ),
    ),
    CorpusChunkSpec(
        key="pg_hnsw_ops",
        section_header="pgvector Query Planner Fallback",
        content=(
            "[Section: pgvector Query Planner Fallback]\n"
            "Catalog entry PGV_ERR_SEQ99: EXPLAIN ANALYZE shows Seq Scan instead of Index Scan on embeddings. "
            "Match operator <=> with vector_cosine_ops, sort directly by distance ASC without arithmetic wrappers, and keep LIMIT."
        ),
    ),
    CorpusChunkSpec(
        key="redis_oom_overview",
        section_header="Redis Memory Pressure Symptoms",
        content=(
            "[Section: Redis Memory Pressure Symptoms]\n"
            "Redis memory pressure occurs when cached payloads, queue metadata, and deduplication entries accumulate "
            "near maxmemory. Clients start seeing write errors while read commands may still succeed."
        ),
    ),
    CorpusChunkSpec(
        key="redis_oom_dont_flush",
        section_header="Redis OOM Safe Handling Warning",
        content=(
            "[Section: Redis OOM Safe Handling Warning]\n"
            "Important warning for Redis OOM incidents: never run FLUSHALL in production when maxmemory is exceeded, "
            "because wiping Redis deletes active ARQ background jobs and in-flight rate limiter counters."
        ),
    ),
    CorpusChunkSpec(
        key="redis_oom_ttl_fix",
        section_header="Redis OOM Deduplication TTL Fix",
        content=(
            "[Section: Redis OOM Deduplication TTL Fix]\n"
            "Root cause fix for NW_RD_OOM77: deduplication keys were written without expiration TTL. "
            "Always set EX 300 on dedup keys and configure Redis maxmemory-policy allkeys-lru."
        ),
    ),
    CorpusChunkSpec(
        key="redis_dedup_race",
        section_header="Redis Alert Storm Race Condition",
        content=(
            "[Section: Redis Alert Storm Race Condition]\n"
            "Catalog entry NW_RD_RACE11: multiple identical incidents inserted within 20ms despite dedup check. "
            "Separate GET then SET introduces a race condition; replace with atomic redis.set(key, val, ex=300, nx=True)."
        ),
    ),
    CorpusChunkSpec(
        key="redis_conn_refused",
        section_header="Redis Network Reachability Failure",
        content=(
            "[Section: Redis Network Reachability Failure]\n"
            "Catalog entry NW_RD_CONN111: ConnectionError 111 connecting to redis:6379. "
            "Check docker compose ps health status and verify REDIS_URL uses localhost outside Docker or redis inside Compose."
        ),
    ),
    CorpusChunkSpec(
        key="arq_overview",
        section_header="ARQ Background Worker Architecture",
        content=(
            "[Section: ARQ Background Worker Architecture]\n"
            "ARQ workers pull queued document ingestion and incident triage jobs from Redis asynchronously "
            "so FastAPI HTTP endpoints can return 202 Accepted within milliseconds."
        ),
    ),
    CorpusChunkSpec(
        key="arq_orphan_commit",
        section_header="ARQ Race Between Enqueue and Commit",
        content=(
            "[Section: ARQ Race Between Enqueue and Commit]\n"
            "Catalog entry NW_ARQ_404: worker logs document not found right after upload. "
            "The API called enqueue_job before await session.commit() made the row visible under READ COMMITTED; "
            "always commit Postgres before pushing to Redis."
        ),
    ),
    CorpusChunkSpec(
        key="arq_loop_block",
        section_header="ARQ Event Loop Freeze on ML Inference",
        content=(
            "[Section: ARQ Event Loop Freeze on ML Inference]\n"
            "Catalog entry NW_ARQ_LOOP88: ARQ worker misses Redis heartbeats and freezes all coroutines during embeddings. "
            "Synchronous PyTorch encode() blocks the event loop; wrap model calls in await asyncio.to_thread(...)."
        ),
    ),
    CorpusChunkSpec(
        key="arq_uuid_serialize",
        section_header="ARQ Job Argument Serialization",
        content=(
            "[Section: ARQ Job Argument Serialization]\n"
            "Catalog entry NW_ARQ_SER19: enqueue_job fails with TypeError UUID is not serializable. "
            "Convert UUID to str(doc.id) when enqueuing and parse uuid.UUID(document_id) inside the worker task."
        ),
    ),
    CorpusChunkSpec(
        key="gemini_429_overview",
        section_header="Gemini Free Tier Quota Limits",
        content=(
            "[Section: Gemini Free Tier Quota Limits]\n"
            "Google Gemini Flash free tier enforces strict requests-per-minute (RPM) and tokens-per-minute limits. "
            "Burst incident traffic can quickly exhaust the shared API quota across workers."
        ),
    ),
    CorpusChunkSpec(
        key="gemini_429_backoff",
        section_header="Gemini 429 Exponential Backoff Fix",
        content=(
            "[Section: Gemini 429 Exponential Backoff Fix]\n"
            "Remediation for HTTP 429 RESOURCE_EXHAUSTED: throttle outgoing calls with a rate limiter and retry "
            "429, 502, 503 errors using exponential backoff with jitter delay = (2 ** attempt) + random.uniform(0, 1)."
        ),
    ),
    CorpusChunkSpec(
        key="gemini_403_no_retry",
        section_header="Gemini Authentication Failure Policy",
        content=(
            "[Section: Gemini Authentication Failure Policy]\n"
            "Catalog entry NW_LLM_AUTH403: HTTP 401 or 403 PERMISSION_DENIED from Google GenAI client. "
            "Never retry 4xx authentication errors; raise immediately and rotate GEMINI_API_KEY in .env."
        ),
    ),
    CorpusChunkSpec(
        key="gemini_loop_guard",
        section_header="AI Agent Infinite Tool Calling Guard",
        content=(
            "[Section: AI Agent Infinite Tool Calling Guard]\n"
            "Catalog entry NW_AGENT_MAX3: LLM investigator keeps calling diagnostic MCP tools endlessly. "
            "Enforce a hard max_iterations=3 loop counter and force structured RootCauseReport output."
        ),
    ),
    CorpusChunkSpec(
        key="gemini_schema_strict",
        section_header="Gemini Structured Output Schema Error",
        content=(
            "[Section: Gemini Structured Output Schema Error]\n"
            "Catalog entry NW_LLM_JSON22: ValidationError parsing RootCauseReport from raw markdown text. "
            "Pass response_mime_type='application/json' and response_schema=RootCauseReport to the Gemini SDK."
        ),
    ),
    CorpusChunkSpec(
        key="fastapi_ctxvar_overview",
        section_header="FastAPI Request ID Middleware Overview",
        content=(
            "[Section: FastAPI Request ID Middleware Overview]\n"
            "FastAPI HTTP middleware attaches a correlation X-Request-ID to every incoming request and response header "
            "so distributed logs and ServiceLog entries can be traced across async coroutines."
        ),
    ),
    CorpusChunkSpec(
        key="fastapi_500_state",
        section_header="FastAPI 500 Handler Missing Request ID",
        content=(
            "[Section: FastAPI 500 Handler Missing Request ID]\n"
            "Fix for missing X-Request-ID inside exception_handler(Exception): ServerErrorMiddleware runs outside "
            "user middleware so ContextVar already reset in finally. Read request.state.request_id instead."
        ),
    ),
    CorpusChunkSpec(
        key="fastapi_perf_counter",
        section_header="FastAPI Latency Clock Skew Bug",
        content=(
            "[Section: FastAPI Latency Clock Skew Bug]\n"
            "Catalog entry NW_API_CLK05: X-Process-Time header returned negative duration after NTP sync. "
            "Replace wall-clock time.time() with monotonic time.perf_counter()."
        ),
    ),
    CorpusChunkSpec(
        key="fastapi_env_extra",
        section_header="Pydantic Settings OS Environment Crash",
        content=(
            "[Section: Pydantic Settings OS Environment Crash]\n"
            "Catalog entry NW_CFG_ENV07: ValidationError extra inputs not permitted for PATH or TEMP on Windows. "
            "Set extra='ignore' inside SettingsConfigDict."
        ),
    ),
    CorpusChunkSpec(
        key="chunker_overlap_guard",
        section_header="Markdown Chunker Infinite Loop Bug",
        content=(
            "[Section: Markdown Chunker Infinite Loop Bug]\n"
            "Catalog entry NW_CHK_OVL14: chunk_markdown consumes 100% RAM when chunk_overlap >= chunk_size "
            "because step becomes zero or negative. Enforce 0 <= chunk_overlap < chunk_size validation."
        ),
    ),
    CorpusChunkSpec(
        key="fts_or_tokens",
        section_header="Postgres FTS Zero Results on Logs",
        content=(
            "[Section: Postgres FTS Zero Results on Logs]\n"
            "Catalog entry NW_FTS_AND08: websearch_to_tsquery returns empty results on noisy tracebacks with IPs "
            "because spaces default to strict AND. Tokenize and join with ' OR ' before searching."
        ),
    ),
    CorpusChunkSpec(
        key="mcp_stderr_only",
        section_header="MCP StdIO Stream Corruption",
        content=(
            "[Section: MCP StdIO Stream Corruption]\n"
            "Catalog entry NW_MCP_STD01: JSON-RPC parse error over stdio transport caused by print() writing to stdout. "
            "Send all diagnostic logs exclusively to sys.stderr."
        ),
    ),
    CorpusChunkSpec(
        key="docker_host_db",
        section_header="Local Alembic Connection Host Error",
        content=(
            "[Section: Local Alembic Connection Host Error]\n"
            "Catalog entry NW_DKR_HOST2: socket.gaierror getaddrinfo failed for host db on Windows host. "
            "Service name db only resolves inside Docker network; use POSTGRES_HOST=localhost in .env."
        ),
    ),
]

EVAL_CASES: list[EvalCase] = [
    EvalCase("Service billing crashed during peak traffic with opaque error SQLSTATE_53300 in logs", "pg_code_53300"),
    EvalCase("Database transaction failed in worker with SQLSTATE_40P01 under concurrent updates", "pg_code_40P01"),
    EvalCase("Heavy analytics query aborted by Postgres with code SQLSTATE_57014", "pg_code_57014"),
    EvalCase("Webhook replay failed with SQLSTATE_23505 on insert", "pg_code_23505"),
    EvalCase("Idle connections failed after DB restart with SQLSTATE_08006", "pg_code_08006"),
    EvalCase("How do I fix QueuePool limit of size 10 overflow 20 connection leaks in Python?", "pg_pool_leak_fix"),
    EvalCase("Which metrics and pg_stat_activity counters show PostgreSQL connection pool wait usage?", "pg_pool_metrics"),
    EvalCase("Session failed after IntegrityError with SA_ERR_PRB01 when updating fallback status", "pg_pending_rollback"),
    EvalCase("Accessing ORM attribute after commit failed with SA_ERR_MGL02", "pg_missing_greenlet"),
    EvalCase("Parallel HNSW index build aborted with PGV_ERR_SHM64 no space left", "pg_hnsw_shm"),
    EvalCase("Vector search ignores HNSW index and runs Seq Scan: PGV_ERR_SEQ99", "pg_hnsw_ops"),
    EvalCase("Why should operators NEVER run FLUSHALL during a Redis OOM incident?", "redis_oom_dont_flush"),
    EvalCase("What is the root cause fix for NW_RD_OOM77 deduplication keys filling Redis memory?", "redis_oom_ttl_fix"),
    EvalCase("Alert storm bypassed deduplication with code NW_RD_RACE11", "redis_dedup_race"),
    EvalCase("Healthcheck failure NW_RD_CONN111 connecting to cache port 6379", "redis_conn_refused"),
    EvalCase("Upload returned 202 but background task logged NW_ARQ_404 immediately", "arq_orphan_commit"),
    EvalCase("Worker froze and missed heartbeats during ingestion with NW_ARQ_LOOP88", "arq_loop_block"),
    EvalCase("Enqueuing job crashed with NW_ARQ_SER19 on document ID", "arq_uuid_serialize"),
    EvalCase("How should we implement retry delay with jitter for Gemini HTTP 429 quota errors?", "gemini_429_backoff"),
    EvalCase("Worker keeps retrying failing GenAI call NW_LLM_AUTH403", "gemini_403_no_retry"),
    EvalCase("Agent burned quota calling tools endlessly: NW_AGENT_MAX3", "gemini_loop_guard"),
    EvalCase("Failed to validate Pydantic report from markdown output: NW_LLM_JSON22", "gemini_schema_strict"),
    EvalCase("Why does unhandled 500 response show dash instead of X-Request-ID from ContextVar?", "fastapi_500_state"),
    EvalCase("Latency header reported negative time: NW_API_CLK05", "fastapi_perf_counter"),
    EvalCase("Local migration failed on Windows with NW_DKR_HOST2", "docker_host_db"),
]


def compute_metrics(rankings: list[tuple[list[uuid.UUID], uuid.UUID]]) -> EvalMetrics:
    if not rankings:
        return EvalMetrics(0.0, 0.0, 0.0)
    hits_3, hits_5, rr_sum = 0, 0, 0.0

    for retrieved_ids, expected_id in rankings:
        if expected_id in retrieved_ids[:3]:
            hits_3 += 1
        if expected_id in retrieved_ids[:5]:
            hits_5 += 1
            rank = retrieved_ids[:5].index(expected_id) + 1
            rr_sum += 1.0 / rank

    return EvalMetrics(
        recall_at_3=hits_3 / len(rankings),
        recall_at_5=hits_5 / len(rankings),
        mrr=rr_sum / len(rankings),
    )


async def seed_eval_corpus(embedder) -> tuple[uuid.UUID, dict[str, uuid.UUID]]:
    texts = [spec.content for spec in CORPUS_SPECS]
    vectors = await embedder.embed_texts(texts)

    key_to_chunk_id: dict[str, uuid.UUID] = {}
    async with async_session_maker() as session:
        doc = Document(
            title="NightWatch RAG Eval Corpus",
            source_type="eval_runbook",
            raw_content="\n\n".join(texts),
            status="READY",
        )
        session.add(doc)
        await session.flush()

        for idx, (spec, vec) in enumerate(zip(CORPUS_SPECS, vectors, strict=True)):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                section_header=spec.section_header,
                content=spec.content,
                embedding=vec,
            )
            session.add(chunk)
            await session.flush()
            key_to_chunk_id[spec.key] = chunk.id

        await session.commit()
        return doc.id, key_to_chunk_id


async def cleanup_eval_corpus(doc_id: uuid.UUID) -> None:
    async with async_session_maker() as session:
        await session.execute(delete(Document).where(Document.id == doc_id))
        await session.commit()


async def main() -> None:
    embedder = await asyncio.to_thread(get_embedding_service)
    reranker = await asyncio.to_thread(get_reranker_service)

    doc_id, key_to_chunk_id = await seed_eval_corpus(embedder)
    try:
        query_texts = [case.query for case in EVAL_CASES]
        query_vectors = await embedder.embed_texts(query_texts)

        vector_rankings: list[tuple[list[uuid.UUID], uuid.UUID]] = []
        hybrid_rankings: list[tuple[list[uuid.UUID], uuid.UUID]] = []
        rerank_rankings: list[tuple[list[uuid.UUID], uuid.UUID]] = []

        async with async_session_maker() as session:
            for case, q_vec in zip(EVAL_CASES, query_vectors, strict=True):
                expected_id = key_to_chunk_id[case.expected_key]

                vec_res = await vector_search(session, q_vec, limit=5)
                vector_rankings.append(([r.chunk_id for r in vec_res], expected_id))

                hyb_top5 = await hybrid_search(
                    session, case.query, q_vec, candidate_limit=20, limit=5
                )
                hybrid_rankings.append(([r.chunk_id for r in hyb_top5], expected_id))

                hyb_candidates = await hybrid_search(
                    session, case.query, q_vec, candidate_limit=20, limit=20
                )
                reranked = await rerank_results(
                    case.query, hyb_candidates, reranker, top_k=5
                )
                rerank_rankings.append(([r.chunk_id for r in reranked], expected_id))

        results = [
            ("Vector only (Top-5)", compute_metrics(vector_rankings)),
            ("Hybrid (Vector + FTS RRF)", compute_metrics(hybrid_rankings)),
            ("Hybrid + Cross-Encoder Rerank", compute_metrics(rerank_rankings)),
        ]

        print(f"\nEvaluated {len(EVAL_CASES)} incidents across {len(CORPUS_SPECS)} runbook chunks:\n")
        print(f"{'Mode':<32} | {'Recall@3':>8} | {'Recall@5':>8} | {'MRR':>8}")
        print("-" * 66)
        for label, metrics in results:
            print(
                f"{label:<32} | {metrics.recall_at_3 * 100:>7.1f}% "
                f"| {metrics.recall_at_5 * 100:>7.1f}% | {metrics.mrr:>8.3f}"
            )
    finally:
        await cleanup_eval_corpus(doc_id)


if __name__ == "__main__":
    asyncio.run(main())
