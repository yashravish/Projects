"""
Sync trigger endpoints.

Allow on-demand triggering of individual or full syncs via HTTP POST.
"""
import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.schemas.sync_job import SyncResult
from app.services.sync_service import execute_crm_sync, execute_full_sync, execute_vendor_sync

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/crm",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger CRM sync",
)
def trigger_crm_sync(db: Session = Depends(get_db)):
    """
    Manually trigger a CRM (JSON) data sync.

    Fetches customers and orders from the mock CRM API, transforms
    them into the normalized schema, and upserts into the database.

    Returns a SyncResult with counts and job ID.
    """
    logger.info("crm_sync_triggered_via_api")
    return execute_crm_sync(db, triggered_by="api")


@router.post(
    "/vendor",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger Vendor sync",
)
def trigger_vendor_sync(db: Session = Depends(get_db)):
    """
    Manually trigger a Vendor (XML) data sync.

    Fetches orders and shipments from the mock Vendor XML API,
    parses XML, transforms, and upserts into the database.

    Returns a SyncResult with counts and job ID.
    """
    logger.info("vendor_sync_triggered_via_api")
    return execute_vendor_sync(db, triggered_by="api")


@router.post(
    "/all",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger full sync (CRM + Vendor)",
)
def trigger_full_sync(db: Session = Depends(get_db)):
    """
    Trigger CRM sync then Vendor sync sequentially and return aggregated results.

    This is the recommended way to perform a complete data refresh.
    """
    logger.info("full_sync_triggered_via_api")
    return execute_full_sync(db, triggered_by="api")
