import logging
import uuid

from arq.connections import RedisSettings
from sqlalchemy import text, update

from app.config import settings
from app.database import async_session_maker, engine
from app.models import Document, DocumentChunk
from app.services.chunker import chunk_markdown

logger = logging.getLogger("nightwatch.worker")


async def startup(ctx: dict) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("ARQ worker started")


async def shutdown(ctx: dict) -> None:
    await engine.dispose()
    logger.info("ARQ worker shut down")


async def ingest_document_task(ctx: dict, document_id: str) -> str:
    doc_uuid = uuid.UUID(document_id)
    async with async_session_maker() as session:
        doc = await session.get(Document, doc_uuid)
        if doc is None:
            logger.warning("Document %s not found", document_id)
            return "not_found"

        doc.status = "PROCESSING"
        await session.commit()

        try:
            chunks = chunk_markdown(doc.raw_content)
            doc_chunks = [DocumentChunk(document_id=doc.id, chunk_index=c.chunk_index, section_header=c.section_header, content=c.content, embedding=None) for c in chunks]
            session.add_all(doc_chunks)
            
            doc.status = "READY"
            await session.commit()
            return "ready"
        except Exception:
            await session.rollback()
            await session.execute(update(Document).where(Document.id == doc_uuid).values(status="FAILED"))
            await session.commit()
            logger.exception("Failed to ingest document %s", document_id)
            raise


class WorkerSettings:
    functions = [ingest_document_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_tries = 3
