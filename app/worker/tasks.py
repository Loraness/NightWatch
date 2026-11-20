import asyncio
import logging
import uuid

from arq.connections import RedisSettings
from sqlalchemy import text, update

from app.config import settings
from app.database import async_session_maker, engine
from app.models import Document, DocumentChunk, Incident
from app.services.chunker import chunk_markdown
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger("nightwatch.worker")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


async def startup(ctx: dict) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Loading local embedding model...")
    ctx["embedder"] = await asyncio.to_thread(get_embedding_service)
    logger.info("ARQ worker started with local embedding model")


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
            embedder: EmbeddingService = ctx.get("embedder") or await asyncio.to_thread(
                get_embedding_service
            )
            vectors = await embedder.embed_texts([c.content for c in chunks])

            doc_chunks = [
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=c.chunk_index,
                    section_header=c.section_header,
                    content=c.content,
                    embedding=vec,
                )
                for c, vec in zip(chunks, vectors, strict=True)
            ]
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


async def investigate_incident_task(ctx: dict, incident_id: str) -> str:
    incident_uuid = uuid.UUID(incident_id)
    async with async_session_maker() as session:
        incident = await session.get(Incident, incident_uuid)
        if incident is None:
            logger.warning("Incident %s not found", incident_id)
            return "not_found"

        incident.status = "INVESTIGATING"
        await session.commit()
        logger.info(
            "Queued incident %s (%s: %s) picked up by worker",
            incident_id,
            incident.service_name,
            incident.exception_type,
        )
        return "investigating"


class WorkerSettings:
    functions = [ingest_document_task, investigate_incident_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_tries = 3
