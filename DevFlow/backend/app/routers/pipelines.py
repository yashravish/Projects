from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.pipeline import PipelineRun, TestResult, TestStatus
from app.models.project import Project
from app.core.database import get_db
from app.schemas.pipelines import (
    PipelineRunRead,
    RecordTestResultBody,
    StageRead,
    TestResultRead,
    TriggerPipelineBody,
)
from app.services.pipeline_service import create_pending_run, get_run_with_relations, run_pipeline_simulation

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.get("/{run_id}", response_model=PipelineRunRead)
async def get_run(run_id: int, session: AsyncSession = Depends(get_db)) -> PipelineRunRead:
    r = await get_run_with_relations(session, run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return _to_read(r)


@router.get("/by-project/{project_id}", response_model=list[PipelineRunRead])
async def list_runs(
    project_id: int, session: AsyncSession = Depends(get_db), limit: int = Query(30, le=200)
) -> list[PipelineRunRead]:
    q = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.project_id == project_id)
        .options(
            selectinload(PipelineRun.stages), selectinload(PipelineRun.test_results)
        )
        .order_by(PipelineRun.id.desc())
        .limit(limit)
    )
    return [_to_read(x) for x in q.scalars().all()]


@router.post("/{project_id}/trigger", response_model=PipelineRunRead, status_code=201)
async def trigger_pipeline(
    project_id: int, body: TriggerPipelineBody, session: AsyncSession = Depends(get_db)
) -> PipelineRunRead:
    p = await session.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    run = await create_pending_run(session, p, body.branch, body.commit_sha)
    await run_pipeline_simulation(session, run, p)
    await session.refresh(run, attribute_names=["stages", "test_results"])
    return _to_read(run)


@router.post("/{run_id}/test-results", response_model=TestResultRead, status_code=201)
async def add_test_result(
    run_id: int, body: RecordTestResultBody, session: AsyncSession = Depends(get_db)
) -> TestResultRead:
    r = await session.get(PipelineRun, run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    try:
        status = TestStatus(body.status)
    except Exception as exc:
        raise HTTPException(400, f"Invalid status: {body.status}") from exc
    tr = TestResult(
        run_id=run_id,
        name=body.name,
        suite=body.suite,
        status=status,
        duration_ms=body.duration_ms,
        message=body.message,
    )
    session.add(tr)
    await session.flush()
    await session.refresh(tr)
    return TestResultRead.model_validate(tr)


def _to_read(r: PipelineRun) -> PipelineRunRead:
    return PipelineRunRead(
        id=r.id,
        project_id=r.project_id,
        status=r.status.value,
        branch=r.branch,
        commit_sha=r.commit_sha,
        started_at=r.started_at,
        finished_at=r.finished_at,
        total_duration_ms=r.total_duration_ms,
        external_ref=r.external_ref,
        stages=[
            StageRead(
                id=s.id,
                name=s.name.value,
                sort_order=s.sort_order,
                status=s.status.value,
                duration_ms=s.duration_ms,
                logs=s.logs,
                passed=s.passed,
            )
            for s in sorted(r.stages, key=lambda x: x.sort_order)
        ],
        test_results=[TestResultRead.model_validate(t) for t in r.test_results],
    )
