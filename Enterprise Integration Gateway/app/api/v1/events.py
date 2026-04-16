"""
Events API endpoints.

Provides a manual event publish endpoint for testing/demo and a status
endpoint showing Kafka consumer connectivity.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, status

from app.core.kafka_client import get_kafka_producer, get_kafka_status
from app.schemas.events import InboundSyncRequestEvent
from app.services.event_publisher import publish_event

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/publish",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Publish a sync request event to Kafka",
)
def publish_sync_request(sync_type: str = "all", requested_by: str = "api"):
    """
    Manually publish an ``InboundSyncRequestEvent`` to the Kafka inbound topic.

    This simulates an external system requesting a sync via message queue.
    Useful for testing and demo purposes.

    - **sync_type**: ``crm``, ``vendor``, or ``all``
    - **requested_by**: identifier of the requesting system
    """
    event = InboundSyncRequestEvent(
        sync_type=sync_type,
        requested_by=requested_by,
        timestamp=datetime.now(timezone.utc),
    )

    producer = get_kafka_producer()
    if producer is None or not producer.is_available:
        return {
            "status": "skipped",
            "message": "Kafka producer is not available. Event was not published.",
            "event": event.to_dict(),
        }

    success = publish_event(event, key=sync_type)
    return {
        "status": "accepted" if success else "failed",
        "message": "Event published to Kafka." if success else "Failed to publish event.",
        "event": event.to_dict(),
    }


@router.get(
    "/status",
    summary="Kafka event system status",
)
def kafka_status():
    """
    Returns Kafka connectivity and configuration status.

    Includes producer availability, configured topics, and bootstrap servers.
    """
    return get_kafka_status()
