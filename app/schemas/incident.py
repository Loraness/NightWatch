import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IncidentWebhookPayload(BaseModel):
    service_name: str = Field(..., min_length=1, max_length=100)
    request_id: str | None = Field(default=None, max_length=64)
    exception_type: str = Field(..., min_length=1, max_length=255)
    error_message: str = Field(..., min_length=1)
    traceback_text: str = Field(..., min_length=1)


class IncidentWebhookResponse(BaseModel):
    status: Literal["queued", "duplicate_ignored"]
    fingerprint: str
    incident_id: uuid.UUID | None = None
    task_id: str | None = None


class IncidentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    fingerprint: str
    service_name: str
    exception_type: str
    status: str
    ai_report: dict | None
    created_at: datetime
