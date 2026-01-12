"""Admin routes for system operations.

These endpoints are restricted to users with ADMIN role in at least one team.
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ForbiddenException, NotFoundException
from app.database import get_db
from app.models import Role, TeamMembership
from app.services.scheduler import JOB_REGISTRY, run_job_by_name

router = APIRouter(prefix="/admin", tags=["admin"])


class JobRunResponse(BaseModel):
    """Response from running a job."""

    job_name: str = Field(..., description="Name of the job that was run")
    status: str = Field(..., description="Result status: 'success' or 'failure'")
    duration_ms: int = Field(..., description="Execution time in milliseconds")
    error: str | None = Field(None, description="Error message if failed")


async def require_admin(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Verify that the current user is an ADMIN in at least one team."""
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.user_id == current_user.id,
            TeamMembership.role == Role.ADMIN,
        )
    )
    admin_membership = result.scalar_one_or_none()

    if admin_membership is None:
        raise ForbiddenException("Admin role required")


@router.post("/jobs/{job_name}/run", response_model=JobRunResponse)
async def run_job(
    job_name: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JobRunResponse:
    """Manually trigger a notification job.

    This endpoint is for testing and operations verification only.
    Uses the same execution path as scheduled runs.

    **ADMIN only**

    Available jobs:
    - no_log_reminder: Daily practice reminder
    - blocking_due_48h: Alert for blocking tickets due soon
    - blocked_over_48h: Alert for long-blocked tickets
    - weekly_leader_digest: Weekly summary for leaders
    """
    # Verify admin access
    await require_admin(current_user, db)

    # Validate job name
    if job_name not in JOB_REGISTRY:
        raise NotFoundException(f"Unknown job: {job_name}")

    # Run the job
    result = await run_job_by_name(job_name)

    return JobRunResponse(
        job_name=result["job_name"],
        status=result["status"],
        duration_ms=result["duration_ms"],
        error=result.get("error"),
    )


@router.get("/jobs")
async def list_jobs(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all available notification jobs.

    **ADMIN only**
    """
    await require_admin(current_user, db)

    return {
        "jobs": [
            {
                "name": name,
                "description": func.__doc__.split("\n")[0] if func.__doc__ else None,
            }
            for name, func in JOB_REGISTRY.items()
        ]
    }

