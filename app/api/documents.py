import uuid

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Document
from app.schemas.document import (
    DocumentAcceptedResponse,
    DocumentCreate,
    DocumentStatusResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


@router.post(
    "",
    response_model=DocumentAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document(
    payload: DocumentCreate,
    session: AsyncSession = Depends(get_db),
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> DocumentAcceptedResponse:

    doc = Document(
        title=payload.title,
        source_type=payload.source_type,
        raw_content=payload.raw_content,
        status="PENDING",

    )
    session.add(doc)
    await session.commit()

    job = await arq_pool.enqueue_job("ingest_document_task", str(doc.id))

    return DocumentAcceptedResponse(
        document_id=doc.id,
        task_id=job.job_id if job else None,
        status=doc.status,
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Document:
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc
