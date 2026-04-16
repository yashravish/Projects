"""
Kafka event publishing service.

Provides a simple ``publish_event()`` function that the sync service calls
at key lifecycle points. If Kafka is unavailable, events are logged but
the caller is never blocked or errored.
"""
import logging
from typing import Optional

from app.core.config import settings
from app.core.kafka_client import get_kafka_producer
from app.schemas.events import BaseEvent

logger = logging.getLogger(__name__)


def publish_event(event: BaseEvent, key: Optional[str] = None) -> bool:
    """
    Serialize and publish an event to the configured Kafka events topic.

    Args:
        event: A Pydantic event instance.
        key: Optional partition key (e.g. correlation_id for ordering).

    Returns:
        True if the event was enqueued, False if Kafka is unavailable.
    """
    producer = get_kafka_producer()
    if producer is None or not producer.is_available:
        logger.debug(
            "event_publish_skipped",
            extra={"event_type": event.event_type, "reason": "producer unavailable"},
        )
        return False

    topic = settings.KAFKA_EVENTS_TOPIC
    event_dict = event.to_dict()
    partition_key = key or event.correlation_id

    success = producer.publish(topic, event_dict, key=partition_key)

    if success:
        logger.info(
            "event_published",
            extra={
                "event_type": event.event_type,
                "topic": topic,
                "correlation_id": event.correlation_id,
            },
        )
    return success
