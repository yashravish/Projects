from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import OrmBase


class SyncJobBase(OrmBase):
    correlation_id: str
    job_type: str
    status: str
    triggered_by: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    error_message: str | None = None
    job_metadata: dict[str, Any] | None = None


class SyncJobResponse(SyncJobBase):
    id: int
    created_at: datetime


class SyncResult(OrmBase):
    """Summary returned immediately after a sync is triggered via API."""

    job_id: int
    correlation_id: str
    job_type: str
    status: str
    records_processed: int
    records_inserted: int
    records_updated: int
    records_failed: int
    message: str = ""
