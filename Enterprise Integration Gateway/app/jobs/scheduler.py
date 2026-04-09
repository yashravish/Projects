"""
APScheduler configuration for background sync jobs.

Uses BackgroundScheduler so jobs run in daemon threads alongside the
FastAPI event loop without blocking it.
"""
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import settings

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler(timezone="UTC")


def start_scheduler() -> None:
    from app.jobs.crm_sync_job import run_crm_sync_job
    from app.jobs.vendor_sync_job import run_vendor_sync_job

    _scheduler.add_job(
        run_crm_sync_job,
        trigger=IntervalTrigger(minutes=settings.CRM_SYNC_INTERVAL_MINUTES),
        id="crm_sync",
        name="CRM Sync",
        replace_existing=True,
        misfire_grace_time=60,
    )
    _scheduler.add_job(
        run_vendor_sync_job,
        trigger=IntervalTrigger(minutes=settings.VENDOR_SYNC_INTERVAL_MINUTES),
        id="vendor_sync",
        name="Vendor Sync",
        replace_existing=True,
        misfire_grace_time=60,
    )

    _scheduler.start()
    logger.info(
        "scheduler_started",
        extra={
            "crm_interval_minutes": settings.CRM_SYNC_INTERVAL_MINUTES,
            "vendor_interval_minutes": settings.VENDOR_SYNC_INTERVAL_MINUTES,
        },
    )


def shutdown_scheduler() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")


def get_scheduler_status() -> dict:
    if not _scheduler.running:
        return {"running": False, "jobs": []}
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
        )
    return {"running": True, "jobs": jobs}
