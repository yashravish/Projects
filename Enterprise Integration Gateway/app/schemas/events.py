"""
Pydantic schemas for Kafka integration events.

All events share a common ``BaseEvent`` parent with ``event_type``,
``timestamp``, and ``correlation_id``. Specialized events add
type-specific payload fields.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Base schema for all integration events."""
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")


class SyncStartedEvent(BaseEvent):
    """Published when a sync job begins execution."""
    event_type: str = "sync.started"
    job_id: int
    job_type: str
    triggered_by: str


class SyncCompletedEvent(BaseEvent):
    """Published when a sync job finishes (success, partial_success, or failed)."""
    event_type: str = "sync.completed"
    job_id: int
    job_type: str
    status: str
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    message: Optional[str] = None


class RecordFailedEvent(BaseEvent):
    """Published when an individual record fails during transformation or upsert."""
    event_type: str = "record.failed"
    job_id: int
    source: str
    record_type: str
    external_id: Optional[str] = None
    error_message: str


class InboundSyncRequestEvent(BaseEvent):
    """
    Consumed from the inbound topic to trigger an async sync.

    Allows external systems to request syncs by publishing messages
    to ``eig.inbound.sync.requests``.
    """
    event_type: str = "sync.request"
    sync_type: str = "all"  # "crm", "vendor", or "all"
    requested_by: str = "kafka"
