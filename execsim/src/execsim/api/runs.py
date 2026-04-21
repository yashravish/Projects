"""API endpoints for simulation runs."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from execsim.config import Settings
from execsim.db.models import Fill, Opportunity, RunStatus, SimulationRun
from execsim.dependencies import get_db, get_settings
from execsim.schemas.common import ErrorResponse, RunStatusEnum
from execsim.schemas.runs import RunCreate, RunDetail, RunResponse, RunSummary
from execsim.simulator.engine import run_simulation
from execsim.simulator.persistence import persist_simulation

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "",
    response_model=RunResponse,
    status_code=201,
    responses={422: {"model": ErrorResponse}},
    summary="Start a simulation run",
)
def create_run(
    body: RunCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RunResponse:
    """Run a simulation with the given seed and persist results."""
    result = run_simulation(seed=body.seed, settings=settings)
    run_id = persist_simulation(db, result)

    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).one()
    return RunResponse(
        id=run.id,
        seed=run.seed,
        status=RunStatusEnum(run.status.value),
        started_at=run.started_at,
    )


@router.get(
    "",
    response_model=list[RunSummary],
    summary="List simulation runs",
)
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: RunStatusEnum | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[RunSummary]:
    """List simulation runs with optional status filter."""
    query = db.query(SimulationRun)
    if status is not None:
        query = query.filter(SimulationRun.status == RunStatus(status.value))
    query = query.order_by(SimulationRun.started_at.desc())
    runs = query.offset(offset).limit(limit).all()
    return [
        RunSummary(
            id=r.id,
            seed=r.seed,
            status=RunStatusEnum(r.status.value),
            num_steps=r.num_steps,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in runs
    ]


@router.get(
    "/{run_id}",
    response_model=RunDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Get run detail",
)
def get_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> RunDetail:
    """Get full detail for a single simulation run."""
    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    num_opps = db.query(Opportunity).filter(Opportunity.run_id == run_id).count()
    num_fills = (
        db.query(Fill)
        .join(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .count()
    )

    return RunDetail(
        id=run.id,
        seed=run.seed,
        status=RunStatusEnum(run.status.value),
        num_steps=run.num_steps,
        config=run.config,
        started_at=run.started_at,
        finished_at=run.finished_at,
        num_opportunities=num_opps,
        num_fills=num_fills,
    )
