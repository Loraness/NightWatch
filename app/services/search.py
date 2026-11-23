import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")


@dataclass(frozen=True)
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    section_header: str | None
    content: str
    score: float


async def vector_search(
    session: AsyncSession,
    query_vector: list[float],
    limit: int = 5,
) -> list[SearchResult]:
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")

    stmt = (
        select(DocumentChunk, distance)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            section_header=chunk.section_header,
            content=chunk.content,
            score=1.0 - float(dist),
        )
        for chunk, dist in rows
    ]


async def fts_search(
    session: AsyncSession,
    query_text: str,
    limit: int = 20,
) -> list[SearchResult]:
    tokens = _TOKEN_RE.findall(query_text)
    if not tokens:
        return []

    or_query = " OR ".join(tokens)
    ts_query = func.websearch_to_tsquery("simple", or_query)
    rank_expr = func.ts_rank_cd(DocumentChunk.search_vector, ts_query).label("rank")

    stmt = (
        select(DocumentChunk, rank_expr)
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank_expr.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            section_header=chunk.section_header,
            content=chunk.content,
            score=float(rank),
        )
        for chunk, rank in rows
    ]


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    fts_results: list[SearchResult],
    k: int = 60,
    limit: int = 20,
) -> list[SearchResult]:
    scores: dict[uuid.UUID, float] = {}
    items: dict[uuid.UUID, SearchResult] = {}

    for rank, res in enumerate(vector_results, start=1):
        items[res.chunk_id] = res 
        scores[res.chunk_id] = scores.get(res.chunk_id, 0.0) + 1.0/(k+rank)

    for rank, res in enumerate(fts_results, start=1):
        items[res.chunk_id] = res 
        scores[res.chunk_id] = scores.get(res.chunk_id, 0.0) + 1.0/(k+rank)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    result = []
    for chunk_id, rrf_score in ranked:
        item = items[chunk_id]
        result.append(
            SearchResult(
                chunk_id=chunk_id,
                document_id=item.document_id,
                section_header=item.section_header,
                content=item.content,
                score=rrf_score
            )
        )
    return result

async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_vector: list[float],
    candidate_limit: int = 20,
    limit: int = 20,
) -> list[SearchResult]:
    vector_results = await vector_search(session, query_vector, limit=candidate_limit)
    fts_results = await fts_search(session, query_text, limit=candidate_limit)
    return reciprocal_rank_fusion(vector_results, fts_results, limit=limit)