"""Calibration drift checker.

Flags when the optimal reserve price on new held-out seeds differs from
the stored optimal by more than a configurable threshold.
"""

from uuid import UUID

from sqlalchemy.orm import Session

from execsim.db.models import (
    Auction,
    ValidationAlert,
    CheckType,
    Severity,
)
from execsim.auction.calibration import calibrate_reserve
from execsim.config import Settings
from execsim.validation.common import create_alert


def run_calibration_check(
    db: Session,
    run_id: UUID,
    stored_reserve_bps: float,
    held_out_seeds: list[int],
    settings: Settings,
    drift_threshold_pct: float = 20.0,
) -> list[ValidationAlert]:
    """Check if calibration has drifted from the stored optimal reserve.

    Args:
        db: SQLAlchemy session.
        run_id: UUID of the run to associate alerts with.
        stored_reserve_bps: Previously calibrated optimal reserve.
        held_out_seeds: New held-out seeds for re-calibration.
        settings: Application settings.
        drift_threshold_pct: Maximum acceptable drift as a percentage. Default 20%.

    Returns:
        List of ValidationAlert objects (not yet committed).
    """
    alerts: list[ValidationAlert] = []

    result = calibrate_reserve(
        held_out_seeds=held_out_seeds,
        n_bidders=settings.auction_n_bidders,
        grid_max_bps=settings.calibration_grid_max_bps,
        grid_step_bps=settings.calibration_grid_step_bps,
        allocation_floor=settings.calibration_floor,
        settings=settings,
    )

    new_reserve = result.optimal_reserve_bps

    if stored_reserve_bps > 0:
        drift_pct = abs(new_reserve - stored_reserve_bps) / stored_reserve_bps * 100.0
    elif new_reserve > 0:
        drift_pct = 100.0  # from zero to non-zero is maximum drift
    else:
        drift_pct = 0.0  # both zero

    if drift_pct > drift_threshold_pct:
        alerts.append(create_alert(
            run_id=run_id,
            check_type=CheckType.calibration,
            severity=Severity.warning,
            message=(
                f"Calibration drift: stored={stored_reserve_bps:.1f} bps, "
                f"new={new_reserve:.1f} bps, drift={drift_pct:.1f}%"
            ),
            detail={
                "stored_reserve_bps": stored_reserve_bps,
                "new_reserve_bps": new_reserve,
                "drift_pct": drift_pct,
                "threshold_pct": drift_threshold_pct,
            },
        ))

    return alerts
