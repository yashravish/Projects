"""Temporal consistency checker.

Flags non-monotone steps or timestamps within a run, and temporal ordering
violations between detection, execution, and run boundaries.
"""

from uuid import UUID

from sqlalchemy.orm import Session

from execsim.db.models import (
    Fill,
    MarketSnapshot,
    Opportunity,
    SimulationRun,
    ValidationAlert,
    CheckType,
    Severity,
)
from execsim.validation.common import create_alert


def run_temporal_check(db: Session, run_id: UUID) -> list[ValidationAlert]:
    """Check temporal consistency for a simulation run.

    Args:
        db: SQLAlchemy session.
        run_id: UUID of the run to validate.

    Returns:
        List of ValidationAlert objects (not yet committed).
    """
    alerts: list[ValidationAlert] = []

    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if run is None:
        return alerts

    # Check snapshot ordering
    snapshots = (
        db.query(MarketSnapshot)
        .filter(MarketSnapshot.run_id == run_id)
        .order_by(MarketSnapshot.step)
        .all()
    )

    for i in range(1, len(snapshots)):
        if snapshots[i].step <= snapshots[i - 1].step:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.temporal,
                severity=Severity.error,
                message=f"Non-monotone step: {snapshots[i-1].step} -> {snapshots[i].step}",
                detail={"index": i, "prev_step": snapshots[i-1].step, "step": snapshots[i].step},
            ))
        if snapshots[i].ts < snapshots[i - 1].ts:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.temporal,
                severity=Severity.error,
                message=f"Non-monotone timestamp at step {snapshots[i].step}",
                detail={"step": snapshots[i].step},
            ))

    # Check opportunity detected_at >= run started_at
    opportunities = db.query(Opportunity).filter(Opportunity.run_id == run_id).all()
    for opp in opportunities:
        if opp.detected_at < run.started_at:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.temporal,
                severity=Severity.error,
                message=f"Opportunity {opp.id}: detected_at < run started_at",
                detail={"opportunity_id": str(opp.id)},
            ))

    # Check fill executed_at >= opportunity detected_at
    for opp in opportunities:
        if opp.fill is not None:
            if opp.fill.executed_at < opp.detected_at:
                alerts.append(create_alert(
                    run_id=run_id,
                    check_type=CheckType.temporal,
                    severity=Severity.error,
                    message=f"Fill {opp.fill.id}: executed_at < detected_at",
                    detail={"fill_id": str(opp.fill.id), "opportunity_id": str(opp.id)},
                ))

    return alerts
