"""
Orchestration layer for integration sync jobs.

Responsibilities:
  - Create SyncJob records (lifecycle: pending → running → terminal)
  - Invoke data-source clients
  - Call transformers
  - Call entity services for persistence
  - Track per-record failures in failed_records
  - Return a SyncResult summary
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.clients.crm_client import CrmClient
from app.clients.vendor_client import VendorClient
from app.core.exceptions import IntegrationError, TransformationError
from app.models.sync_job import SyncJob
from app.schemas.sync_job import SyncResult
from app.services.customer_service import upsert_customer
from app.services.failed_record_service import create_failed_record
from app.services.order_service import upsert_order
from app.services.shipment_service import upsert_shipment
from app.utils.correlation import new_correlation_id
from app.utils.transformers import (
    transform_crm_customer,
    transform_crm_order,
    transform_vendor_order,
    transform_vendor_shipment,
)
from app.utils.xml_parser import parse_vendor_orders, parse_vendor_shipments

logger = logging.getLogger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _create_job(db: Session, job_type: str, triggered_by: str) -> SyncJob:
    job = SyncJob(
        correlation_id=new_correlation_id(),
        job_type=job_type,
        status="running",
        triggered_by=triggered_by,
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(
        "sync_job_started",
        extra={"job_id": job.id, "job_type": job_type, "correlation_id": job.correlation_id},
    )
    return job


def _finish_job(
    db: Session,
    job: SyncJob,
    *,
    processed: int,
    inserted: int,
    updated: int,
    failed: int,
    error_message: str | None = None,
) -> SyncJob:
    if failed == 0 and error_message is None:
        status = "success"
    elif failed > 0 and (inserted + updated) > 0:
        status = "partial_success"
    elif failed > 0 and (inserted + updated) == 0:
        status = "failed"
    else:
        status = "success"

    job.status = status
    job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    job.records_processed = processed
    job.records_inserted = inserted
    job.records_updated = updated
    job.records_failed = failed
    job.error_message = error_message
    db.commit()
    db.refresh(job)
    logger.info(
        "sync_job_finished",
        extra={
            "job_id": job.id,
            "status": status,
            "processed": processed,
            "inserted": inserted,
            "updated": updated,
            "failed": failed,
        },
    )
    return job


def _fail_job(db: Session, job: SyncJob, error_message: str) -> SyncJob:
    job.status = "failed"
    job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    job.error_message = error_message
    db.commit()
    db.refresh(job)
    logger.error(
        "sync_job_failed",
        extra={"job_id": job.id, "error": error_message},
    )
    return job


# ── CRM Sync ──────────────────────────────────────────────────────────────────


def execute_crm_sync(db: Session, triggered_by: str = "api") -> SyncResult:
    """
    Full CRM sync flow:
      1. Fetch JSON customers + orders
      2. Transform each record
      3. Upsert into DB
      4. Track failures in failed_records
    """
    job = _create_job(db, "crm_sync", triggered_by)
    inserted = updated = failed = 0

    # ── Map: external_id → internal customer.id for order FK linking ──────────
    customer_id_map: dict[str, int] = {}

    try:
        with CrmClient() as client:
            raw_customers = client.get_customers()
            raw_orders = client.get_orders()
    except IntegrationError as exc:
        return SyncResult(
            job_id=job.id,
            correlation_id=job.correlation_id,
            job_type=job.job_type,
            status=_fail_job(db, job, str(exc)).status,
            records_processed=0,
            records_inserted=0,
            records_updated=0,
            records_failed=0,
            message=str(exc),
        )

    # ── Process customers ──────────────────────────────────────────────────────
    for raw in raw_customers:
        try:
            schema = transform_crm_customer(raw)
            customer, created = upsert_customer(db, schema)
            db.commit()
            customer_id_map[schema.external_id] = customer.id
            if created:
                inserted += 1
            else:
                updated += 1
        except (TransformationError, Exception) as exc:
            failed += 1
            db.rollback()
            create_failed_record(
                db,
                sync_job_id=job.id,
                source="crm",
                record_type="customer",
                external_id=raw.get("customerId") or raw.get("id"),
                raw_data=json.dumps(raw),
                error_message=str(exc),
            )
            db.commit()
            logger.warning("crm_customer_failed", extra={"error": str(exc)})

    # ── Process orders ─────────────────────────────────────────────────────────
    for raw in raw_orders:
        try:
            crm_customer_id = raw.get("customerId")
            internal_customer_id = customer_id_map.get(str(crm_customer_id)) if crm_customer_id else None
            schema = transform_crm_order(raw, customer_id=internal_customer_id)
            _, created = upsert_order(db, schema)
            db.commit()
            if created:
                inserted += 1
            else:
                updated += 1
        except (TransformationError, Exception) as exc:
            failed += 1
            db.rollback()
            create_failed_record(
                db,
                sync_job_id=job.id,
                source="crm",
                record_type="order",
                external_id=raw.get("orderId") or raw.get("id"),
                raw_data=json.dumps(raw),
                error_message=str(exc),
            )
            db.commit()
            logger.warning("crm_order_failed", extra={"error": str(exc)})

    total = len(raw_customers) + len(raw_orders)
    job = _finish_job(
        db, job,
        processed=total,
        inserted=inserted,
        updated=updated,
        failed=failed,
    )
    return SyncResult(
        job_id=job.id,
        correlation_id=job.correlation_id,
        job_type=job.job_type,
        status=job.status,
        records_processed=job.records_processed,
        records_inserted=job.records_inserted,
        records_updated=job.records_updated,
        records_failed=job.records_failed,
        message=f"CRM sync completed. {inserted} inserted, {updated} updated, {failed} failed.",
    )


# ── Vendor Sync ───────────────────────────────────────────────────────────────


def execute_vendor_sync(db: Session, triggered_by: str = "api") -> SyncResult:
    """
    Full Vendor XML sync flow:
      1. Fetch XML orders + shipments
      2. Parse XML safely (captures malformed records)
      3. Transform valid records
      4. Upsert into DB
    """
    job = _create_job(db, "vendor_sync", triggered_by)
    inserted = updated = failed = 0

    # ── Map: vendor_order_id → internal order.id for shipment FK ─────────────
    order_id_map: dict[str, int] = {}

    try:
        with VendorClient() as client:
            orders_xml = client.get_orders_xml()
            shipments_xml = client.get_shipments_xml()
    except IntegrationError as exc:
        return SyncResult(
            job_id=job.id,
            correlation_id=job.correlation_id,
            job_type=job.job_type,
            status=_fail_job(db, job, str(exc)).status,
            records_processed=0,
            records_inserted=0,
            records_updated=0,
            records_failed=0,
            message=str(exc),
        )

    # ── Process vendor orders ─────────────────────────────────────────────────
    valid_orders, malformed_orders = parse_vendor_orders(orders_xml)

    for record in malformed_orders:
        failed += 1
        create_failed_record(
            db,
            sync_job_id=job.id,
            source="vendor",
            record_type="order",
            external_id=None,
            raw_data=record.get("raw", ""),
            error_message=record.get("error", "XML parse failure"),
        )
        db.commit()

    for parsed in valid_orders:
        try:
            schema = transform_vendor_order(parsed)
            order, created = upsert_order(db, schema)
            db.commit()
            order_id_map[parsed["order_id"]] = order.id
            if created:
                inserted += 1
            else:
                updated += 1
        except (TransformationError, Exception) as exc:
            failed += 1
            db.rollback()
            create_failed_record(
                db,
                sync_job_id=job.id,
                source="vendor",
                record_type="order",
                external_id=parsed.get("order_id"),
                raw_data=parsed.get("raw_xml", ""),
                error_message=str(exc),
            )
            db.commit()
            logger.warning("vendor_order_failed", extra={"error": str(exc)})

    # ── Process vendor shipments ──────────────────────────────────────────────
    valid_shipments, malformed_shipments = parse_vendor_shipments(shipments_xml)

    for record in malformed_shipments:
        failed += 1
        create_failed_record(
            db,
            sync_job_id=job.id,
            source="vendor",
            record_type="shipment",
            external_id=None,
            raw_data=record.get("raw", ""),
            error_message=record.get("error", "XML parse failure"),
        )
        db.commit()

    for parsed in valid_shipments:
        try:
            vendor_order_ref = parsed.get("vendor_order_id")
            internal_order_id = order_id_map.get(vendor_order_ref) if vendor_order_ref else None
            schema = transform_vendor_shipment(parsed, order_id=internal_order_id)
            _, created = upsert_shipment(db, schema)
            db.commit()
            if created:
                inserted += 1
            else:
                updated += 1
        except (TransformationError, Exception) as exc:
            failed += 1
            db.rollback()
            create_failed_record(
                db,
                sync_job_id=job.id,
                source="vendor",
                record_type="shipment",
                external_id=parsed.get("shipment_id"),
                raw_data=parsed.get("raw_xml", ""),
                error_message=str(exc),
            )
            db.commit()
            logger.warning("vendor_shipment_failed", extra={"error": str(exc)})

    total = len(valid_orders) + len(malformed_orders) + len(valid_shipments) + len(malformed_shipments)
    job = _finish_job(
        db, job,
        processed=total,
        inserted=inserted,
        updated=updated,
        failed=failed,
    )
    return SyncResult(
        job_id=job.id,
        correlation_id=job.correlation_id,
        job_type=job.job_type,
        status=job.status,
        records_processed=job.records_processed,
        records_inserted=job.records_inserted,
        records_updated=job.records_updated,
        records_failed=job.records_failed,
        message=f"Vendor sync completed. {inserted} inserted, {updated} updated, {failed} failed.",
    )


# ── Full Sync ─────────────────────────────────────────────────────────────────


def execute_full_sync(db: Session, triggered_by: str = "api") -> SyncResult:
    """
    Trigger CRM sync then Vendor sync, aggregate results.
    Creates an overarching 'full_sync' SyncJob in addition to individual jobs.
    """
    job = _create_job(db, "full_sync", triggered_by)

    crm_result = execute_crm_sync(db, triggered_by=triggered_by)
    vendor_result = execute_vendor_sync(db, triggered_by=triggered_by)

    total_processed = crm_result.records_processed + vendor_result.records_processed
    total_inserted = crm_result.records_inserted + vendor_result.records_inserted
    total_updated = crm_result.records_updated + vendor_result.records_updated
    total_failed = crm_result.records_failed + vendor_result.records_failed

    job = _finish_job(
        db, job,
        processed=total_processed,
        inserted=total_inserted,
        updated=total_updated,
        failed=total_failed,
    )

    return SyncResult(
        job_id=job.id,
        correlation_id=job.correlation_id,
        job_type=job.job_type,
        status=job.status,
        records_processed=total_processed,
        records_inserted=total_inserted,
        records_updated=total_updated,
        records_failed=total_failed,
        message=(
            f"Full sync completed. CRM: {crm_result.status}. "
            f"Vendor: {vendor_result.status}. "
            f"{total_inserted} inserted, {total_updated} updated, {total_failed} failed."
        ),
    )


# ── Job query helpers ─────────────────────────────────────────────────────────


def list_sync_jobs(
    db: Session,
    job_type: str | None = None,
    status: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[SyncJob], int]:
    from sqlalchemy import select, desc
    stmt = select(SyncJob).order_by(desc(SyncJob.created_at))
    if job_type:
        stmt = stmt.where(SyncJob.job_type == job_type)
    if status:
        stmt = stmt.where(SyncJob.status == status)
    all_rows = db.scalars(stmt).all()
    return list(all_rows[skip: skip + limit]), len(all_rows)


def get_sync_job(db: Session, job_id: int) -> SyncJob | None:
    return db.get(SyncJob, job_id)
