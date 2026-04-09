from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.core.exceptions import RecordNotFoundError, RetryExhaustedError
from app.schemas.failed_record import FailedRecordResponse, RetryResponse
from app.services.failed_record_service import list_failed_records, retry_failed_record

router = APIRouter()


@router.get("", response_model=list[FailedRecordResponse], summary="List failed records")
def get_failed_records(
    source: str | None = Query(None, description="Filter by source: 'crm' or 'vendor'"),
    record_type: str | None = Query(None, description="Filter by type: 'customer', 'order', 'shipment'"),
    status: str | None = Query(None, description="Filter by status: pending_retry, retrying, resolved, abandoned"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Return records that failed during a sync run.

    These are entries in the dead-letter queue that require investigation
    and/or manual retry.
    """
    records, _ = list_failed_records(
        db, source=source, status=status, record_type=record_type, skip=skip, limit=limit
    )
    return records


@router.post(
    "/{record_id}/retry",
    response_model=RetryResponse,
    summary="Retry a failed record",
)
def retry_record(record_id: int, db: Session = Depends(get_db)):
    """
    Attempt to re-process a single failed record.

    The original raw payload is re-run through the transformation
    and persistence pipeline. On success the record is marked 'resolved'.

    A record will be marked 'abandoned' after exceeding MAX_RETRY_COUNT
    and further retry calls will be rejected.
    """
    try:
        return retry_failed_record(db, record_id)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except RetryExhaustedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
