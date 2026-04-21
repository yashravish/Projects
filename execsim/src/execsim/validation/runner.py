"""Validation runner: orchestrates all validation checks for a run."""

from uuid import UUID

from sqlalchemy.orm import Session

from execsim.db.models import ValidationAlert
from execsim.validation.schema_check import run_schema_check
from execsim.validation.temporal_check import run_temporal_check
from execsim.validation.state_check import run_state_check


def run_all_checks(db: Session, run_id: UUID) -> list[ValidationAlert]:
    """Run all validation checks for a simulation run.

    Runs schema, temporal, and state checks. Calibration check is excluded
    because it requires additional parameters (stored reserve, held-out seeds)
    and is triggered separately.

    Args:
        db: SQLAlchemy session.
        run_id: UUID of the run to validate.

    Returns:
        List of ValidationAlert objects. These are persisted by the caller.
    """
    alerts: list[ValidationAlert] = []
    alerts.extend(run_schema_check(db, run_id))
    alerts.extend(run_temporal_check(db, run_id))
    alerts.extend(run_state_check(db, run_id))

    # Persist alerts
    for alert in alerts:
        db.add(alert)
    db.commit()

    return alerts
