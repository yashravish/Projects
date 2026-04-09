from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.schemas.sync_job import SyncJobResponse
from app.services.sync_service import get_sync_job, list_sync_jobs

router = APIRouter()


@router.get("", response_model=list[SyncJobResponse], summary="List integration jobs")
def get_integration_jobs(
    job_type: str | None = Query(None, description="Filter by type: crm_sync, vendor_sync, full_sync"),
    status: str | None = Query(None, description="Filter by status: pending, running, success, partial_success, failed"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    Return the history of integration sync jobs, most recent first.

    Use this endpoint to audit past runs and investigate failures.
    """
    jobs, _ = list_sync_jobs(db, job_type=job_type, status=status, skip=skip, limit=limit)
    return jobs


@router.get("/{job_id}", response_model=SyncJobResponse, summary="Get integration job by ID")
def get_integration_job(job_id: int, db: Session = Depends(get_db)):
    """Return details for a specific integration job including error messages."""
    job = get_sync_job(db, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Integration job {job_id} not found",
        )
    return job
