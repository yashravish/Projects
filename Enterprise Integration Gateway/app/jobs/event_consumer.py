"""
Background Kafka consumer for inbound sync requests.

Listens on the ``eig.inbound.sync.requests`` topic and triggers syncs
when valid ``InboundSyncRequestEvent`` messages arrive. Runs in a daemon
thread alongside the FastAPI event loop (same pattern as APScheduler).
"""
import logging
from typing import Optional

from app.core.config import settings
from app.core.kafka_client import KafkaEventConsumer

logger = logging.getLogger(__name__)

_consumer: Optional[KafkaEventConsumer] = None


def _handle_inbound_sync(message: dict) -> None:
    """
    Process an inbound sync request message.

    Delegates to the sync service based on ``sync_type``.
    Opens its own DB session (same pattern as scheduler jobs).
    """
    from app.db.session import SessionLocal
    from app.services.sync_service import (
        execute_crm_sync,
        execute_full_sync,
        execute_vendor_sync,
    )

    sync_type = message.get("sync_type", "all")
    requested_by = message.get("requested_by", "kafka")

    logger.info(
        "inbound_sync_request",
        extra={"sync_type": sync_type, "requested_by": requested_by},
    )

    db = SessionLocal()
    try:
        if sync_type == "crm":
            result = execute_crm_sync(db, triggered_by=f"kafka:{requested_by}")
        elif sync_type == "vendor":
            result = execute_vendor_sync(db, triggered_by=f"kafka:{requested_by}")
        else:
            result = execute_full_sync(db, triggered_by=f"kafka:{requested_by}")

        logger.info(
            "inbound_sync_completed",
            extra={
                "sync_type": sync_type,
                "status": result.status,
                "records_processed": result.records_processed,
            },
        )
    except Exception as exc:
        logger.error(
            "inbound_sync_failed",
            extra={"sync_type": sync_type, "error": str(exc)},
        )
    finally:
        db.close()


def start_event_consumer() -> Optional[KafkaEventConsumer]:
    """
    Start the inbound sync request consumer in a background thread.

    Called during application startup if ``KAFKA_ENABLED=true``.
    """
    global _consumer

    if not settings.KAFKA_ENABLED:
        logger.info("event_consumer_not_started", extra={"reason": "KAFKA_ENABLED=false"})
        return None

    _consumer = KafkaEventConsumer(
        topics=[settings.KAFKA_INBOUND_TOPIC],
        handler=_handle_inbound_sync,
    )
    _consumer.start()
    return _consumer


def stop_event_consumer() -> None:
    """Stop the inbound sync request consumer."""
    global _consumer
    if _consumer is not None:
        _consumer.stop()
        _consumer = None
