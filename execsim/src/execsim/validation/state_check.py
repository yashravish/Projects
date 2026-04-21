"""State consistency checker.

Flags impossible states: AMM reserves negative, execution prices wildly
outside the bid-ask range, fills on a venue at a step with no book data.
"""

from uuid import UUID

from sqlalchemy.orm import Session

from execsim.db.models import (
    Fill,
    MarketSnapshot,
    Opportunity,
    ValidationAlert,
    CheckType,
    Severity,
)
from execsim.validation.common import create_alert


def run_state_check(db: Session, run_id: UUID) -> list[ValidationAlert]:
    """Check state consistency for a simulation run.

    Args:
        db: SQLAlchemy session.
        run_id: UUID of the run to validate.

    Returns:
        List of ValidationAlert objects (not yet committed).
    """
    alerts: list[ValidationAlert] = []

    # Build snapshot lookup by step
    snapshots = (
        db.query(MarketSnapshot)
        .filter(MarketSnapshot.run_id == run_id)
        .all()
    )
    snap_by_step = {s.step: s for s in snapshots}

    # Check AMM reserves
    for snap in snapshots:
        if snap.amm_reserve_x <= 0 or snap.amm_reserve_y <= 0:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.state,
                severity=Severity.error,
                message=f"Step {snap.step}: AMM reserves non-positive (x={snap.amm_reserve_x}, y={snap.amm_reserve_y})",
                detail={"step": snap.step, "amm_x": snap.amm_reserve_x, "amm_y": snap.amm_reserve_y},
            ))

    # Check fill exec_prices are reasonable
    fills = (
        db.query(Fill)
        .join(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .all()
    )

    for fill in fills:
        opp = fill.opportunity
        step = opp.step
        snap = snap_by_step.get(step)

        if snap is None:
            alerts.append(create_alert(
                run_id=run_id,
                check_type=CheckType.state,
                severity=Severity.error,
                message=f"Fill {fill.id}: no snapshot at step {step}",
                detail={"fill_id": str(fill.id), "step": step},
            ))
            continue

        # Determine reasonable price range based on venue
        if fill.venue.value in ("venue_a", "venue_b"):
            if fill.venue.value == "venue_a":
                bid = snap.venue_a_bid
                ask = snap.venue_a_ask
            else:
                bid = snap.venue_b_bid
                ask = snap.venue_b_ask

            spread = ask - bid
            lower = bid - 5 * spread
            upper = ask + 5 * spread

            if fill.exec_price < lower or fill.exec_price > upper:
                alerts.append(create_alert(
                    run_id=run_id,
                    check_type=CheckType.state,
                    severity=Severity.warning,
                    message=(
                        f"Fill {fill.id}: exec_price {fill.exec_price:.4f} outside "
                        f"[bid-5*spread, ask+5*spread] = [{lower:.4f}, {upper:.4f}]"
                    ),
                    detail={
                        "fill_id": str(fill.id),
                        "exec_price": fill.exec_price,
                        "lower": lower,
                        "upper": upper,
                    },
                ))

    return alerts
