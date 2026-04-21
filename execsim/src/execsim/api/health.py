"""Health check endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from execsim.dependencies import get_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
def health_check() -> HealthResponse:
    """Liveness check. Returns 200 if the API process is running."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
)
def readiness_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Readiness check. Returns 200 if the database is reachable."""
    db.execute(text("SELECT 1"))
    return HealthResponse(status="ready")
