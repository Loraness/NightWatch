import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    source_type: Literal["runbook", "postmortem"] = "runbook"
    raw_content: str = Field(..., min_length=1)


class DocumentAcceptedResponse(BaseModel):
    document_id: uuid.UUID
    task_id: str | None
    status: str


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_type: str
    status: str
    created_at: datetime
