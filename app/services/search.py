import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk


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
    return [SearchResult(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        section_header=chunk.section_header,
        content=chunk.content,
        score=1.0-dist,
    ) for chunk, dist in rows]