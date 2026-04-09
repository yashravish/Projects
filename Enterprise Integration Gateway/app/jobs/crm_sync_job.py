"""Scheduled CRM sync job — executed by APScheduler."""
import logging

from app.db.session import SessionLocal
from app.services.sync_service import execute_crm_sync

logger = logging.getLogger(__name__)


def run_crm_sync_job() -> None:
    """Entry point for the scheduled CRM sync."""
    logger.info("scheduled_crm_sync_starting")
    db = SessionLocal()
    try:
        result = execute_crm_sync(db, triggered_by="scheduler")
        logger.info(
            "scheduled_crm_sync_finished",
            extra={
                "job_id": result.job_id,
                "status": result.status,
                "inserted": result.records_inserted,
                "updated": result.records_updated,
                "failed": result.records_failed,
            },
        )
    except Exception as exc:
        logger.error("scheduled_crm_sync_error", exc_info=exc)
    finally:
        db.close()
