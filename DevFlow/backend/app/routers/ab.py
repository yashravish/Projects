import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.ab_experiment import ABExperiment, ABMetricRollup, ExperimentStatus
from app.models.project import Project
from app.schemas.ab import (
    ABAssignBody,
    ABExperimentCreate,
    ABExperimentRead,
    ABMetricIngestBody,
    ABMetricRollupRead,
)

router = APIRouter(prefix="/api/experiments", tags=["a-b"])


@router.get("/by-project/{project_id}", response_model=list[ABExperimentRead])
async def list_experiments(
    project_id: int, session: AsyncSession = Depends(get_db), limit: int = Query(50, le=200)
) -> list[ABExperimentRead]:
    r = await session.execute(
        select(ABExperiment)
        .where(ABExperiment.project_id == project_id)
        .options(selectinload(ABExperiment.rollups))
        .order_by(ABExperiment.id.desc())
        .limit(limit)
    )
    return [_to_read(e) for e in r.scalars().all()]


@router.get("/{exp_id}/aggregate", response_model=dict)
async def aggregate_exp(exp_id: int, session: AsyncSession = Depends(get_db)) -> dict:
    e = await session.execute(
        select(ABExperiment)
        .options(selectinload(ABExperiment.rollups))
        .where(ABExperiment.id == exp_id)
    )
    ex = e.scalar_one_or_none()
    if not ex:
        raise HTTPException(404, "Experiment not found")
    rows: list[dict] = []
    for r in ex.rollups:
        conv = (r.conversion_count / r.assignments) if r.assignments else 0.0
        lat = (r.sum_latency_ms / r.assignments) if r.assignments else 0.0
        errr = (r.error_count / r.assignments) if r.assignments else 0.0
        rows.append(
            {
                "variant": r.variant,
                "assignments": r.assignments,
                "conversion_rate": round(conv, 4),
                "avg_latency_ms": round(lat, 2),
                "error_rate": round(errr, 4),
            }
        )
    return {"experiment_id": ex.id, "key": ex.key, "metrics": rows}


@router.post("", response_model=ABExperimentRead, status_code=201)
async def create_experiment(
    body: ABExperimentCreate, session: AsyncSession = Depends(get_db)
) -> ABExperimentRead:
    p = await session.get(Project, body.project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    e = ABExperiment(
        project_id=body.project_id,
        key=body.key,
        name=body.name,
        variant_a_name=body.variant_a_name,
        variant_b_name=body.variant_b_name,
        traffic_a_percent=body.traffic_a_percent,
        key_metric=body.key_metric,
        notes=body.notes,
        status=ExperimentStatus.draft,
    )
    session.add(e)
    await session.flush()
    for v in ("A", "B"):
        session.add(ABMetricRollup(experiment_id=e.id, variant=v))
    await session.flush()
    await session.refresh(e, attribute_names=["rollups"])
    return _to_read(e)


@router.post("/assign", response_model=dict)
async def assign(body: ABAssignBody, session: AsyncSession = Depends(get_db)) -> dict:
    e = await session.get(ABExperiment, body.experiment_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    from app.services.hash_util import stable_variant_choice

    v = stable_variant_choice(body.user_id, e.traffic_a_percent)
    v_name = e.variant_a_name if v == "A" else e.variant_b_name
    return {
        "experiment_id": e.id,
        "user_id": body.user_id,
        "variant": v,
        "variant_name": v_name,
    }


@router.post("/metrics", response_model=dict)
async def ingest_metric(body: ABMetricIngestBody, session: AsyncSession = Depends(get_db)) -> dict:
    e = await session.get(ABExperiment, body.experiment_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    from app.services.hash_util import stable_variant_choice

    expected = stable_variant_choice(body.user_id, e.traffic_a_percent)
    if body.variant not in (expected, "A", "B") and body.variant not in (e.variant_a_name, e.variant_b_name):
        pass
    vkey = "A" if body.variant in ("A", e.variant_a_name) else "B"
    r = await session.execute(
        select(ABMetricRollup).where(
            ABMetricRollup.experiment_id == e.id, ABMetricRollup.variant == vkey
        )
    )
    roll = r.scalar_one_or_none()
    if not roll:
        roll = ABMetricRollup(experiment_id=e.id, variant=vkey)
        session.add(roll)
        await session.flush()
    roll.assignments = roll.assignments + 1
    if body.conversion:
        roll.conversion_count = roll.conversion_count + 1
    roll.sum_latency_ms = roll.sum_latency_ms + max(0.0, body.latency_ms)
    if body.error:
        roll.error_count = roll.error_count + 1
    roll.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    await session.flush()
    return {
        "ok": True,
        "variant": vkey,
        "assignments": roll.assignments,
    }


def _to_read(e: ABExperiment) -> ABExperimentRead:
    return ABExperimentRead(
        id=e.id,
        project_id=e.project_id,
        key=e.key,
        name=e.name,
        variant_a_name=e.variant_a_name,
        variant_b_name=e.variant_b_name,
        traffic_a_percent=e.traffic_a_percent,
        status=e.status.value,
        key_metric=e.key_metric,
        notes=e.notes,
        created_at=e.created_at,
        updated_at=e.updated_at,
        rollups=[ABMetricRollupRead.model_validate(r) for r in e.rollups],
    )
