"""
Health check endpoint.

Verifies:
  - Application is alive
  - Database connectivity
  - Redis connectivity
  - Kafka producer availability
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_db
from app.core.redis_client import get_redis_status
from app.core.kafka_client import get_kafka_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", summary="Health check")
def health_check(db: Session = Depends(get_db)):
    """
    Returns service health including database, Redis, and Kafka connectivity.

    - **status**: 'healthy' or 'degraded'
    - **database**: 'ok' or error message
    - **redis**: connectivity and memory info
    - **kafka**: producer availability and topic config
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

    redis_info = get_redis_status()
    kafka_info = get_kafka_status()

    # Overall status — healthy only if database is ok (Redis/Kafka are optional)
    overall = "healthy" if db_status == "ok" else "degraded"

    payload = {
        "status": overall,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "checks": {
            "database": db_status,
            "redis": redis_info,
            "kafka": kafka_info,
        },
    }
    if db_error:
        payload["checks"]["database_error"] = db_error

    return payload
