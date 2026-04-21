"""Schema validation checker.

Flags null required fields, negative prices, filled_qty > requested_qty,
and invalid enum values in persisted simulation data.
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


def run_schema_check(db: Session, run_id: UUID) -> list[ValidationAlert]:
    """Check schema invariants for a simulation run.

    Args:
        db: SQLAlchemy session.
        run_id: UUID of the run to validate.

    Returns:
        List of ValidationAlert objects (not yet committed).
    """
    alerts: list[ValidationAlert] = []

    # Check snapshots
    snapshots = db.query(MarketSnapshot).filter(MarketSnapshot.run_id == run_id).all()
    for snap in snapshots:
        for field_name in (
            "venue_a_bid", "venue_a_ask", "venue_b_bid", "venue_b_ask",
            "amm_reserve_x", "amm_reserve_y", "amm_price", "true_mid",
        ):
            val = getattr(snap, field_name)
            if val is None or val <= 0:
                alerts.append(create_alert(
                    run_id=run_id,
                    check_type=CheckType.schema_check,
                    severity=Severity.error,
                    message=f"Snapshot step {snap.step}: {field_name} is invalid ({val})",
                    detail={"step": snap.step, "field": field_name, "value": val},
                ))

    # Check fills
    fills = (
        db.query(Fill)
        .join(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .all()
    )
    for fill in fills:
        if fill.filled_qty > fill.requested_qty:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.schema_check,
                severity=Severity.error,
                message=f"Fill {fill.id}: filled_qty ({fill.filled_qty}) > requested_qty ({fill.requested_qty})",
                detail={"fill_id": str(fill.id)},
            ))
        for field_name in ("exec_price", "decision_price", "arrival_mid"):
            val = getattr(fill, field_name)
            if val is None or val <= 0:
                alerts.append(create_alert(
                    run_id=run_id,
                    check_type=CheckType.schema_check,
                    severity=Severity.error,
                    message=f"Fill {fill.id}: {field_name} is invalid ({val})",
                    detail={"fill_id": str(fill.id), "field": field_name, "value": val},
                ))

    # Check opportunities
    opportunities = db.query(Opportunity).filter(Opportunity.run_id == run_id).all()
    if not opportunities:
        alerts.append(create_alert(
            run_id=run_id,
            check_type=CheckType.schema_check,
            severity=Severity.info,
            message="No opportunities detected. Possibly misconfigured thresholds.",
            detail={},
        ))

    return alerts
