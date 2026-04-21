"""Persistence helpers for saving simulation results to the database."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from execsim.db.models import (
    CheckType,
    ExecutionMetric,
    Fill,
    MarketSnapshot,
    Opportunity,
    OpportunityType,
    RunStatus,
    Side,
    SimulationRun,
    Venue,
)
from execsim.simulator.engine import SimulationResult


def persist_simulation(db: Session, result: SimulationResult) -> uuid.UUID:
    """Save a complete simulation result to the database.

    Args:
        db: SQLAlchemy session.
        result: SimulationResult from the simulation engine.

    Returns:
        UUID of the created SimulationRun row.
    """
    run = SimulationRun(
        id=uuid.uuid4(),
        seed=result.seed,
        config=result.config,
        started_at=result.started_at,
        finished_at=result.finished_at,
        status=RunStatus.completed,
        num_steps=result.num_steps,
    )
    db.add(run)

    # Persist snapshots
    for snap in result.snapshots:
        db.add(MarketSnapshot(
            id=uuid.uuid4(),
            run_id=run.id,
            step=snap.step,
            venue_a_bid=snap.venue_a_bid,
            venue_a_ask=snap.venue_a_ask,
            venue_b_bid=snap.venue_b_bid,
            venue_b_ask=snap.venue_b_ask,
            amm_reserve_x=snap.amm_reserve_x,
            amm_reserve_y=snap.amm_reserve_y,
            amm_price=snap.amm_price,
            true_mid=snap.true_mid,
            ts=snap.ts,
            has_liquidation=snap.has_liquidation,
        ))

    # Map opportunity records to DB objects, keyed by in-memory id
    opp_id_map: dict[uuid.UUID, uuid.UUID] = {}
    for opp_rec in result.opportunities:
        db_opp_id = uuid.uuid4()
        opp_id_map[opp_rec.id] = db_opp_id
        db.add(Opportunity(
            id=db_opp_id,
            run_id=run.id,
            step=opp_rec.step,
            type=OpportunityType(opp_rec.kind),
            side=Side(opp_rec.side),
            estimated_value_bps=opp_rec.estimated_value_bps,
            edge_bps=opp_rec.edge_bps,
            arrival_mid=opp_rec.arrival_mid,
            detail=opp_rec.detail,
            detected_at=opp_rec.detected_at,
        ))

    # Persist fills and metrics
    for i, fill_rec in enumerate(result.fills):
        db_opp_id = opp_id_map[fill_rec.opportunity_id]
        fill_db_id = uuid.uuid4()
        db.add(Fill(
            id=fill_db_id,
            opportunity_id=db_opp_id,
            venue=Venue(fill_rec.venue),
            requested_qty=fill_rec.requested_qty,
            filled_qty=fill_rec.filled_qty,
            exec_price=fill_rec.exec_price,
            decision_price=fill_rec.decision_price,
            arrival_mid=fill_rec.arrival_mid,
            latency_steps=fill_rec.latency_steps,
            executed_at=fill_rec.executed_at,
        ))

        if i < len(result.metrics):
            metric = result.metrics[i]
            db.add(ExecutionMetric(
                id=uuid.uuid4(),
                fill_id=fill_db_id,
                impl_shortfall_bps=metric.impl_shortfall_bps,
                realized_slippage_bps=metric.realized_slippage_bps,
                fill_quality=metric.fill_quality,
            ))

    db.commit()
    return run.id
