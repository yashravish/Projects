"""Scheduled Vendor sync job — executed by APScheduler."""
import logging

from app.db.session import SessionLocal
from app.services.sync_service import execute_vendor_sync

logger = logging.getLogger(__name__)


def run_vendor_sync_job() -> None:
    """Entry point for the scheduled Vendor sync."""
    logger.info("scheduled_vendor_sync_starting")
    db = SessionLocal()
    try:
        result = execute_vendor_sync(db, triggered_by="scheduler")
        logger.info(
            "scheduled_vendor_sync_finished",
            extra={
                "job_id": result.job_id,
                "status": result.status,
                "inserted": result.records_inserted,
                "updated": result.records_updated,
                "failed": result.records_failed,
            },
        )
    except Exception as exc:
        logger.error("scheduled_vendor_sync_error", exc_info=exc)
    finally:
        db.close()
