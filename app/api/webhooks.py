import hashlib

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Incident
from app.schemas.incident import IncidentWebhookPayload, IncidentWebhookResponse

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

DEDUP_TTL_SECONDS = 300


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


def _extract_error_location(traceback_text: str) -> str:
    lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith('File "'):
            return line
    return lines[0] if lines else ""


def compute_fingerprint(payload: IncidentWebhookPayload) -> str:
    location = _extract_error_location(payload.traceback_text)
    raw = f"{payload.service_name}:{payload.exception_type}:{location}".lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@router.post(
    "/incident",
    response_model=IncidentWebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def receive_incident_webhook(
    payload: IncidentWebhookPayload,
    session: AsyncSession = Depends(get_db),
    redis: ArqRedis = Depends(get_arq_pool),
) -> IncidentWebhookResponse:
    fingerprint = compute_fingerprint(payload)
    dedup_key = f"dedup:incident:{fingerprint}"

    is_new = await redis.set(dedup_key, fingerprint, ex=DEDUP_TTL_SECONDS, nx=True)

    if not is_new:
        return IncidentWebhookResponse(status="duplicate_ignored", fingerprint=fingerprint)

    incident = Incident(**payload.model_dump(), fingerprint=fingerprint, status="QUEUED")

    session.add(incident)
    await session.commit()

    job = await redis.enqueue_job("investigate_incident_task", str(incident.id))
    return IncidentWebhookResponse(
        status="queued",
        fingerprint=fingerprint,
        incident_id=incident.id,
        task_id=job.job_id if job else None,
    )
