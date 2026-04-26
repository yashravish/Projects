import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deployment import Deployment, DeploymentStatus
from app.models.project import Project
from app.core.database import get_db
from app.schemas.deployments import (
    CanaryBody,
    DeploymentCreateBody,
    DeploymentRead,
    RollbackBody,
)
from app.services.deployment_service import run_canary_simulation, rollback_deployment
from app.services.metrics_state import global_metrics

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


@router.get("/by-project/{project_id}", response_model=list[DeploymentRead])
async def list_deployments(
    project_id: int, session: AsyncSession = Depends(get_db), limit: int = Query(30, le=200)
) -> list[DeploymentRead]:
    q = await session.execute(
        select(Deployment)
        .where(Deployment.project_id == project_id)
        .order_by(Deployment.id.desc())
        .limit(limit)
    )
    return [DeploymentRead.model_validate(d) for d in q.scalars().all()]


@router.post("/{project_id}", response_model=DeploymentRead, status_code=201)
async def create_deployment(
    project_id: int, body: DeploymentCreateBody, session: AsyncSession = Depends(get_db)
) -> DeploymentRead:
    p = await session.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    d = Deployment(
        project_id=project_id,
        version=body.version,
        status=DeploymentStatus.pending,
        environment=body.environment,
        canary_percent=0,
        error_rate=0.0,
    )
    session.add(d)
    await session.flush()
    if body.canary:
        await run_canary_simulation(session, d, target_max_percent=body.canary_start_percent)
    else:
        d.status = DeploymentStatus.healthy
        d.canary_percent = 100
        d.error_rate = 0.01
        d.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        global_metrics.record_deployment(True)
    await session.refresh(d)
    return DeploymentRead.model_validate(d)


@router.post("/{deploy_id}/canary", response_model=DeploymentRead)
async def canary(
    deploy_id: int, body: CanaryBody, session: AsyncSession = Depends(get_db)
) -> DeploymentRead:
    d = await session.get(Deployment, deploy_id)
    if not d:
        raise HTTPException(404, "Deployment not found")
    d = await run_canary_simulation(session, d, target_max_percent=body.target_max_percent)
    return DeploymentRead.model_validate(d)


@router.post("/{deploy_id}/rollback", response_model=DeploymentRead)
async def rollback(
    deploy_id: int, body: RollbackBody, session: AsyncSession = Depends(get_db)
) -> DeploymentRead:
    d = await session.get(Deployment, deploy_id)
    if not d:
        raise HTTPException(404, "Deployment not found")
    prev: Deployment | None = None
    if d.rolled_back_from_id:
        prev = await session.get(Deployment, d.rolled_back_from_id)
    d = await rollback_deployment(session, d, previous=prev)
    return DeploymentRead.model_validate(d)
