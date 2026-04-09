"""
Service for managing dead-letter / failed-record entries.
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import RecordNotFoundError, RetryExhaustedError, TransformationError
from app.models.failed_record import FailedRecord
from app.schemas.failed_record import RetryResponse
from app.utils.transformers import (
    transform_crm_customer,
    transform_crm_order,
    transform_vendor_order,
    transform_vendor_shipment,
)
from app.utils.xml_parser import parse_vendor_orders, parse_vendor_shipments

logger = logging.getLogger(__name__)


def create_failed_record(
    db: Session,
    *,
    sync_job_id: int | None,
    source: str,
    record_type: str,
    external_id: str | None,
    raw_data: str,
    error_message: str,
) -> FailedRecord:
    record = FailedRecord(
        sync_job_id=sync_job_id,
        source=source,
        record_type=record_type,
        external_id=external_id,
        raw_data=raw_data,
        error_message=error_message,
        status="pending_retry",
        retry_count=0,
    )
    db.add(record)
    db.flush()
    logger.info(
        "failed_record_created",
        extra={
            "record_id": record.id,
            "source": source,
            "record_type": record_type,
            "external_id": external_id,
        },
    )
    return record


def list_failed_records(
    db: Session,
    source: str | None = None,
    status: str | None = None,
    record_type: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[FailedRecord], int]:
    stmt = select(FailedRecord)
    if source:
        stmt = stmt.where(FailedRecord.source == source)
    if status:
        stmt = stmt.where(FailedRecord.status == status)
    if record_type:
        stmt = stmt.where(FailedRecord.record_type == record_type)
    all_rows = db.scalars(stmt).all()
    return list(all_rows[skip: skip + limit]), len(all_rows)


def retry_failed_record(db: Session, record_id: int) -> RetryResponse:
    """
    Attempt to re-process a failed record.

    - Increments retry_count.
    - Re-runs transformation.
    - On success, marks as 'resolved'.
    - On failure, marks as 'pending_retry' (or 'abandoned' if max retries exceeded).
    """
    record = db.get(FailedRecord, record_id)
    if record is None:
        raise RecordNotFoundError("FailedRecord", record_id)

    if record.retry_count >= settings.MAX_RETRY_COUNT:
        record.status = "abandoned"
        db.flush()
        raise RetryExhaustedError(record_id, record.retry_count)

    record.status = "retrying"
    record.retry_count += 1
    record.last_retried_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    try:
        _execute_retry(db, record)
        record.status = "resolved"
        db.commit()
        logger.info("failed_record_resolved", extra={"record_id": record_id})
        return RetryResponse(
            record_id=record_id,
            status="resolved",
            retry_count=record.retry_count,
            message="Record successfully re-processed",
        )
    except (TransformationError, ValueError) as exc:
        db.rollback()
        # Re-load record after rollback so we can update its status
        record = db.get(FailedRecord, record_id)
        record.status = "pending_retry"
        record.error_message = str(exc)
        db.commit()
        logger.warning(
            "failed_record_retry_failed",
            extra={"record_id": record_id, "error": str(exc)},
        )
        return RetryResponse(
            record_id=record_id,
            status="pending_retry",
            retry_count=record.retry_count,
            message=f"Retry failed: {exc}",
        )


def _execute_retry(db: Session, record: FailedRecord) -> None:
    """
    Re-run the transformation for a failed record based on its source and type.
    On success, upsert into the appropriate table.
    """
    from app.services.customer_service import upsert_customer
    from app.services.order_service import upsert_order
    from app.services.shipment_service import upsert_shipment

    raw = record.raw_data or ""

    if record.source == "crm":
        raw_dict = json.loads(raw)
        if record.record_type == "customer":
            schema = transform_crm_customer(raw_dict)
            upsert_customer(db, schema)
        elif record.record_type == "order":
            schema = transform_crm_order(raw_dict)
            upsert_order(db, schema)
    elif record.source == "vendor":
        if record.record_type == "order":
            valid, _ = parse_vendor_orders(raw)
            if not valid:
                raise TransformationError("order", record.external_id, "Could not parse vendor XML on retry")
            schema = transform_vendor_order(valid[0])
            upsert_order(db, schema)
        elif record.record_type == "shipment":
            valid, _ = parse_vendor_shipments(raw)
            if not valid:
                raise TransformationError("shipment", record.external_id, "Could not parse vendor XML on retry")
            schema = transform_vendor_shipment(valid[0])
            upsert_shipment(db, schema)
