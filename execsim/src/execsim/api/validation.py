"""API endpoints for validation alerts."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from execsim.db.models import SimulationRun, ValidationAlert
from execsim.dependencies import get_db
from execsim.schemas.common import CheckTypeEnum, ErrorResponse, SeverityEnum
from execsim.schemas.validation import ValidationAlertSchema
from execsim.validation.runner import run_all_checks

router = APIRouter(tags=["validation"])


@router.post(
    "/runs/{run_id}/validate",
    response_model=list[ValidationAlertSchema],
    responses={404: {"model": ErrorResponse}},
    summary="Run validation on a stored run",
)
def validate_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[ValidationAlertSchema]:
    """Run all validation checks on a stored simulation run."""
    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if run is None:
        raise HTTPException(404, detail=f"Run {run_id} not found")

    alerts = run_all_checks(db, run_id)
    return [
        ValidationAlertSchema(
            id=a.id,
            run_id=a.run_id,
            check_type=CheckTypeEnum(a.check_type.value),
            severity=SeverityEnum(a.severity.value),
            message=a.message,
            detail=a.detail,
            created_at=a.created_at,
        )
        for a in alerts
    ]


@router.get(
    "/alerts",
    response_model=list[ValidationAlertSchema],
    summary="List all validation alerts",
)
def list_alerts(
    run_id: uuid.UUID | None = Query(default=None),
    check_type: CheckTypeEnum | None = Query(default=None),
    severity: SeverityEnum | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ValidationAlertSchema]:
    """List validation alerts with optional filters."""
    query = db.query(ValidationAlert)
    if run_id is not None:
        query = query.filter(ValidationAlert.run_id == run_id)
    if check_type is not None:
        query = query.filter(ValidationAlert.check_type == check_type.value)
    if severity is not None:
        query = query.filter(ValidationAlert.severity == severity.value)
    query = query.order_by(ValidationAlert.created_at.desc())
    alerts = query.offset(offset).limit(limit).all()
    return [
        ValidationAlertSchema(
            id=a.id,
            run_id=a.run_id,
            check_type=CheckTypeEnum(a.check_type.value),
            severity=SeverityEnum(a.severity.value),
            message=a.message,
            detail=a.detail,
            created_at=a.created_at,
        )
        for a in alerts
    ]
