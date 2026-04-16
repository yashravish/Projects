from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import (
    User, Vendor, Assessment, Finding, RemediationItem, ControlDomain, AuditLog,
)
from backend.schemas import DashboardStats

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardStats)
def get_dashboard(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    total_vendors = db.query(Vendor).count()
    active_assessments = db.query(Assessment).filter(
        Assessment.status.in_(["draft", "in_progress"])
    ).count()

    open_critical = db.query(Finding).filter(
        Finding.severity == "Critical",
        Finding.remediation_status.in_(["open", "in_progress"]),
    ).count()
    open_high = db.query(Finding).filter(
        Finding.severity == "High",
        Finding.remediation_status.in_(["open", "in_progress"]),
    ).count()

    overdue = db.query(RemediationItem).filter(
        RemediationItem.status.in_(["open", "in_progress"]),
        RemediationItem.due_date < date.today(),
    ).count()

    cat_counts = dict(
        db.query(Vendor.category, func.count(Vendor.id))
        .group_by(Vendor.category).all()
    )

    sev_counts = dict(
        db.query(Finding.severity, func.count(Finding.id))
        .filter(Finding.remediation_status.in_(["open", "in_progress"]))
        .group_by(Finding.severity).all()
    )

    domain_rows = (
        db.query(ControlDomain.name, func.count(Finding.id))
        .join(Finding, Finding.control_domain_id == ControlDomain.id)
        .filter(Finding.remediation_status.in_(["open", "in_progress"]))
        .group_by(ControlDomain.name)
        .all()
    )
    domain_counts = dict(domain_rows)

    recent = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(10)
        .all()
    )
    recent_activity = [
        {
            "action": r.action,
            "entity_type": r.entity_type,
            "entity_id": r.entity_id,
            "details": r.details,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in recent
    ]

    return DashboardStats(
        total_vendors=total_vendors,
        active_assessments=active_assessments,
        open_critical_findings=open_critical,
        open_high_findings=open_high,
        overdue_remediations=overdue,
        vendors_by_category=cat_counts,
        findings_by_severity=sev_counts,
        findings_by_domain=domain_counts,
        recent_activity=recent_activity,
    )
