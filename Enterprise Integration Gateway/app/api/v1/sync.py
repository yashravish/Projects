"""
Sync trigger endpoints.

Allow on-demand triggering of individual or full syncs via HTTP POST.
Rate-limited via Redis sliding window. Invalidates cached responses after sync.
"""
import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.cache import invalidate_cache
from app.core.dependencies import get_db
from app.core.rate_limiter import sync_rate_limiter
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
def trigger_crm_sync(
    db: Session = Depends(get_db),
    _rate_limit=Depends(sync_rate_limiter),
):
    """
    Manually trigger a CRM (JSON) data sync.

    Fetches customers and orders from the mock CRM API, transforms
    them into the normalized schema, and upserts into the database.

    Returns a SyncResult with counts and job ID.

    **Rate limited** — max requests per minute configured via ``RATE_LIMIT_RPM``.
    """
    logger.info("crm_sync_triggered_via_api")
    result = execute_crm_sync(db, triggered_by="api")
    invalidate_cache("customers", "orders", "metrics")
    return result


@router.post(
    "/vendor",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger Vendor sync",
)
def trigger_vendor_sync(
    db: Session = Depends(get_db),
    _rate_limit=Depends(sync_rate_limiter),
):
    """
    Manually trigger a Vendor (XML) data sync.

    Fetches orders and shipments from the mock Vendor XML API,
    parses XML, transforms, and upserts into the database.

    Returns a SyncResult with counts and job ID.

    **Rate limited** — max requests per minute configured via ``RATE_LIMIT_RPM``.
    """
    logger.info("vendor_sync_triggered_via_api")
    result = execute_vendor_sync(db, triggered_by="api")
    invalidate_cache("orders", "shipments", "metrics")
    return result


@router.post(
    "/all",
    response_model=SyncResult,
    status_code=status.HTTP_200_OK,
    summary="Trigger full sync (CRM + Vendor)",
)
def trigger_full_sync(
    db: Session = Depends(get_db),
    _rate_limit=Depends(sync_rate_limiter),
):
    """
    Trigger CRM sync then Vendor sync sequentially and return aggregated results.

    This is the recommended way to perform a complete data refresh.

    **Rate limited** — max requests per minute configured via ``RATE_LIMIT_RPM``.
    """
    logger.info("full_sync_triggered_via_api")
    result = execute_full_sync(db, triggered_by="api")
    invalidate_cache("customers", "orders", "shipments", "metrics")
    return result
