"""API endpoints for opportunities, fills, and metrics."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from execsim.db.models import ExecutionMetric, Fill, Opportunity
from execsim.dependencies import get_db
from execsim.schemas.common import ErrorResponse, OpportunityTypeEnum
from execsim.schemas.fills import FillDetail
from execsim.schemas.metrics import ExecutionMetricSchema, RunMetrics
from execsim.schemas.opportunities import OpportunityDetail

router = APIRouter(tags=["opportunities"])


@router.get(
    "/runs/{run_id}/opportunities",
    response_model=list[OpportunityDetail],
    responses={404: {"model": ErrorResponse}},
    summary="List opportunities for a run",
)
def list_opportunities(
    run_id: uuid.UUID,
    type: OpportunityTypeEnum | None = Query(default=None),
    min_value_bps: float | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
) -> list[OpportunityDetail]:
    """List opportunities for a simulation run with optional filters."""
    query = db.query(Opportunity).filter(Opportunity.run_id == run_id)
    if type is not None:
        query = query.filter(Opportunity.type == type.value)
    if min_value_bps is not None:
        query = query.filter(Opportunity.estimated_value_bps >= min_value_bps)
    query = query.order_by(Opportunity.step)
    opps = query.all()
    return [
        OpportunityDetail(
            id=o.id,
            run_id=o.run_id,
            step=o.step,
            type=OpportunityTypeEnum(o.type.value),
            side=o.side.value,
            estimated_value_bps=o.estimated_value_bps,
            arrival_mid=o.arrival_mid,
            edge_bps=o.edge_bps,
            detail=o.detail,
            detected_at=o.detected_at,
        )
        for o in opps
    ]


@router.get(
    "/runs/{run_id}/fills",
    response_model=list[FillDetail],
    summary="List fills for a run",
)
def list_fills(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[FillDetail]:
    """List all fills for a simulation run."""
    fills = (
        db.query(Fill)
        .join(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .order_by(Fill.executed_at)
        .all()
    )
    return [
        FillDetail(
            id=f.id,
            opportunity_id=f.opportunity_id,
            venue=f.venue.value,
            requested_qty=f.requested_qty,
            filled_qty=f.filled_qty,
            exec_price=f.exec_price,
            decision_price=f.decision_price,
            arrival_mid=f.arrival_mid,
            latency_steps=f.latency_steps,
            executed_at=f.executed_at,
        )
        for f in fills
    ]


@router.get(
    "/runs/{run_id}/metrics",
    response_model=RunMetrics,
    summary="Aggregate metrics for a run",
)
def get_run_metrics(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> RunMetrics:
    """Compute and return aggregate execution metrics for a run."""
    num_opps = db.query(Opportunity).filter(Opportunity.run_id == run_id).count()
    metrics = (
        db.query(ExecutionMetric)
        .join(Fill)
        .join(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .all()
    )

    if not metrics:
        return RunMetrics(
            run_id=run_id,
            num_opportunities=num_opps,
            num_fills=0,
        )

    total_edge = (
        db.query(Opportunity.edge_bps)
        .filter(Opportunity.run_id == run_id)
        .all()
    )

    return RunMetrics(
        run_id=run_id,
        num_opportunities=num_opps,
        num_fills=len(metrics),
        mean_impl_shortfall_bps=sum(m.impl_shortfall_bps for m in metrics) / len(metrics),
        mean_realized_slippage_bps=sum(m.realized_slippage_bps for m in metrics) / len(metrics),
        mean_fill_quality=sum(m.fill_quality for m in metrics) / len(metrics),
        total_edge_bps=sum(e[0] for e in total_edge),
    )
