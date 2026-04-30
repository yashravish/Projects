"""Reporting routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.report import (
    ApprovalSummaryReport,
    TasksByPriorityReport,
    TasksByStatusReport,
    UserWorkloadReport,
)
from app.services.report_service import ReportService


router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/tasks-by-status", response_model=TasksByStatusReport)
def tasks_by_status(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> TasksByStatusReport:
    return ReportService(db).tasks_by_status()


@router.get("/tasks-by-priority", response_model=TasksByPriorityReport)
def tasks_by_priority(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> TasksByPriorityReport:
    return ReportService(db).tasks_by_priority()


@router.get("/user-workload", response_model=UserWorkloadReport)
def user_workload(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> UserWorkloadReport:
    return ReportService(db).user_workload()


@router.get("/approval-summary", response_model=ApprovalSummaryReport)
def approval_summary(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> ApprovalSummaryReport:
    return ReportService(db).approval_summary()


@router.get("/export/tasks.csv")
def export_tasks_csv(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> Response:
    csv_content = ReportService(db).export_tasks_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tasks.csv"},
    )
