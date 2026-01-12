"""Health check endpoint.

GET /health - Returns application health status with DB connectivity check

This endpoint:
- Requires no authentication
- Returns real DB connectivity status
- Returns 503 if critical components are down
- Does not expose sensitive data
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
async def health_check(
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Return application health status with DB connectivity check.

    This endpoint:
    - Is unauthenticated (for load balancer/orchestrator health checks)
    - Performs a real DB connectivity test
    - Returns "ok" or "error" for DB status
    - Returns HTTP 200 if healthy, HTTP 503 if unhealthy
    - Does not expose sensitive information
    """
    db_status = "ok"

    try:
        # Simple DB connectivity check
        await db.execute(text("SELECT 1"))
        logger.debug("health_check_success", db="ok")
    except Exception as e:
        logger.error("health_check_db_error", error=str(e), error_type=type(e).__name__)
        db_status = "error"

    # Overall status is "ok" only if all components are healthy
    overall_status = "ok" if db_status == "ok" else "degraded"

    # Return 503 if critical components are down
    # This tells Railway/load balancers that the service is not ready
    if overall_status == "degraded":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("health_check_degraded", status=overall_status, db=db_status)

    return HealthResponse(
        status=overall_status,
        db=db_status,
    )
