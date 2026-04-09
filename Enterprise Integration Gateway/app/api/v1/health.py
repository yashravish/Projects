"""
Health check endpoint.

Verifies:
  - Application is alive
  - Database connectivity
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", summary="Health check")
def health_check(db: Session = Depends(get_db)):
    """
    Returns service health including database connectivity.

    - **status**: 'healthy' or 'degraded'
    - **database**: 'ok' or error message
    - **version**: application version
    """
    db_status = "ok"
    db_error = None
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = "error"
        db_error = str(exc)
        logger.error("health_check_db_error", exc_info=exc)

    overall = "healthy" if db_status == "ok" else "degraded"

    payload = {
        "status": overall,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "checks": {
            "database": db_status,
        },
    }
    if db_error:
        payload["checks"]["database_error"] = db_error

    return payload
