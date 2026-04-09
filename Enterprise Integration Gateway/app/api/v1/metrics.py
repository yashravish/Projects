"""
Admin / metrics status endpoint.

Provides an operational dashboard view: record counts, latest sync jobs,
failed record summary, and scheduler state.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.jobs.scheduler import get_scheduler_status
from app.models.customer import Customer
from app.models.failed_record import FailedRecord
from app.models.order import Order
from app.models.shipment import Shipment
from app.models.sync_job import SyncJob

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admin/status", summary="Admin status dashboard")
def admin_status(db: Session = Depends(get_db)):
    """
    Returns an operational snapshot:
    - total record counts per entity
    - latest 5 sync jobs
    - failed record counts by status
    - scheduler state
    """
    total_customers = db.scalar(select(func.count(Customer.id))) or 0
    total_orders = db.scalar(select(func.count(Order.id))) or 0
    total_shipments = db.scalar(select(func.count(Shipment.id))) or 0

    failed_pending = (
        db.scalar(
            select(func.count(FailedRecord.id)).where(FailedRecord.status == "pending_retry")
        ) or 0
    )
    failed_abandoned = (
        db.scalar(
            select(func.count(FailedRecord.id)).where(FailedRecord.status == "abandoned")
        ) or 0
    )
    failed_resolved = (
        db.scalar(
            select(func.count(FailedRecord.id)).where(FailedRecord.status == "resolved")
        ) or 0
    )

    recent_jobs_stmt = select(SyncJob).order_by(SyncJob.created_at.desc()).limit(5)
    recent_jobs = db.scalars(recent_jobs_stmt).all()

    recent_jobs_data = [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "triggered_by": j.triggered_by,
            "records_processed": j.records_processed,
            "records_failed": j.records_failed,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in recent_jobs
    ]

    return {
        "record_counts": {
            "customers": total_customers,
            "orders": total_orders,
            "shipments": total_shipments,
        },
        "failed_records": {
            "pending_retry": failed_pending,
            "abandoned": failed_abandoned,
            "resolved": failed_resolved,
        },
        "recent_sync_jobs": recent_jobs_data,
        "scheduler": get_scheduler_status(),
    }


@router.get("/metrics", summary="Basic metrics endpoint")
def metrics(db: Session = Depends(get_db)):
    """
    Lightweight Prometheus-style text metrics (counts only).
    Suitable for a simple health dashboard scrape.
    """
    total_customers = db.scalar(select(func.count(Customer.id))) or 0
    total_orders = db.scalar(select(func.count(Order.id))) or 0
    total_shipments = db.scalar(select(func.count(Shipment.id))) or 0
    total_failed = db.scalar(select(func.count(FailedRecord.id))) or 0
    total_jobs = db.scalar(select(func.count(SyncJob.id))) or 0

    return {
        "eig_customers_total": total_customers,
        "eig_orders_total": total_orders,
        "eig_shipments_total": total_shipments,
        "eig_failed_records_total": total_failed,
        "eig_sync_jobs_total": total_jobs,
    }
