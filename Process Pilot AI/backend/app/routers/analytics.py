from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.auth.jwt_handler import get_current_user
from app.database import get_db
from app.models.request import Request
from app.models.user import User
from app.schemas.analytics import (
    AnalyticsOverview,
    CategoryCount,
    DepartmentCount,
    PainPoint,
    PriorityCount,
    StatusCount,
)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

CLOSED_STATUSES = ("resolved", "closed")


@router.get("/overview", response_model=AnalyticsOverview)
def get_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total = db.query(func.count(Request.id)).scalar() or 0
    closed = (
        db.query(func.count(Request.id))
        .filter(Request.status.in_(CLOSED_STATUSES))
        .scalar()
        or 0
    )
    open_count = total - closed
    avg_priority = (
        db.query(func.coalesce(func.avg(Request.priority_score), 0.0)).scalar()
    )
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    this_week = (
        db.query(func.count(Request.id))
        .filter(Request.created_at >= week_ago)
        .scalar()
        or 0
    )
    return AnalyticsOverview(
        total_requests=total,
        open_requests=open_count,
        closed_requests=closed,
        avg_priority=round(float(avg_priority), 1),
        requests_this_week=this_week,
    )


@router.get("/by-category", response_model=list[CategoryCount])
def by_category(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Request.category, func.count(Request.id))
        .group_by(Request.category)
        .all()
    )
    return [CategoryCount(category=r[0], count=r[1]) for r in rows]


@router.get("/by-department", response_model=list[DepartmentCount])
def by_department(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(User.department, func.count(Request.id))
        .join(User, Request.requester_id == User.id)
        .group_by(User.department)
        .all()
    )
    return [DepartmentCount(department=r[0], count=r[1]) for r in rows]


@router.get("/by-priority", response_model=list[PriorityCount])
def by_priority(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    priority_bucket = case(
        (Request.priority_score.is_(None), "Low (0-3.9)"),
        (Request.priority_score < 4, "Low (0-3.9)"),
        (Request.priority_score < 7, "Medium (4-6.9)"),
        (Request.priority_score < 9, "High (7-8.9)"),
        else_="Critical (9-10)",
    )
    rows = (
        db.query(priority_bucket, func.count(Request.id))
        .group_by(priority_bucket)
        .all()
    )
    return [PriorityCount(priority_range=r[0], count=r[1]) for r in rows]


@router.get("/status-distribution", response_model=list[StatusCount])
def status_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Request.status, func.count(Request.id))
        .group_by(Request.status)
        .all()
    )
    return [StatusCount(status=r[0], count=r[1]) for r in rows]


@router.get("/top-pain-points", response_model=list[PainPoint])
def top_pain_points(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Request)
        .filter(Request.priority_score.isnot(None))
        .order_by(Request.priority_score.desc())
        .limit(5)
        .all()
    )
    return [
        PainPoint(
            description=r.title,
            count=int(r.priority_score or 0),
            category=r.category,
        )
        for r in rows
    ]
