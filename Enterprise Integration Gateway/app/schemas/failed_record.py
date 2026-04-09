from datetime import datetime

from app.schemas.common import OrmBase


class FailedRecordBase(OrmBase):
    sync_job_id: int | None = None
    source: str
    record_type: str
    external_id: str | None = None
    raw_data: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    status: str = "pending_retry"
    last_retried_at: datetime | None = None


class FailedRecordResponse(FailedRecordBase):
    id: int
    created_at: datetime
    updated_at: datetime


class RetryResponse(OrmBase):
    record_id: int
    status: str
    retry_count: int
    message: str
