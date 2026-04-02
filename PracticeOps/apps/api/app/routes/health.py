"""Health check endpoints.

GET /health  - Liveness check for Railway/load balancers (always 200 if process is up)
GET /readyz  - Readiness check with DB connectivity (returns 503 if DB is down)

These endpoints:
- Require no authentication
- Do not expose sensitive data
"""

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database import get_db

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    db: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe: confirms the process is running.

    Always returns HTTP 200. Railway uses this to decide whether the
    deployment succeeded. DB connectivity is checked separately at /readyz.
    """
    db_status = "unknown"

    try:
        from app.database import async_session_maker

        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
            db_status = "ok"
    except Exception:
        db_status = "degraded"
        logger.warning("health_check_db_unreachable")

    return HealthResponse(status="ok", db=db_status)


@router.get("/readyz", response_model=HealthResponse)
async def readiness_check(
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Readiness probe: confirms the app can serve traffic (DB is reachable).

    Returns HTTP 503 when the database is unreachable.
    """
    db_status = "ok"

    try:
        await db.execute(text("SELECT 1"))
        logger.debug("readiness_check_success", db="ok")
    except Exception as e:
        logger.error("readiness_check_db_error", error=str(e), error_type=type(e).__name__)
        db_status = "error"

    overall_status = "ok" if db_status == "ok" else "degraded"

    if overall_status == "degraded":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("readiness_check_degraded", status=overall_status, db=db_status)

    return HealthResponse(status=overall_status, db=db_status)
