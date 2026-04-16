from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, Finding, ControlDomain, Assessment, Vendor
from backend.schemas import FindingOut

router = APIRouter(prefix="/api/findings", tags=["findings"])


@router.get("", response_model=list[FindingOut])
def list_findings(
    severity: str | None = Query(None),
    status: str | None = Query(None),
    domain: str | None = Query(None),
    vendor_id: int | None = Query(None),
    assessment_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Finding)

    if assessment_id:
        q = q.filter(Finding.assessment_id == assessment_id)
    if vendor_id:
        assessment_ids = [a.id for a in db.query(Assessment).filter(Assessment.vendor_id == vendor_id).all()]
        q = q.filter(Finding.assessment_id.in_(assessment_ids))
    if severity:
        q = q.filter(Finding.severity == severity)
    if status:
        q = q.filter(Finding.remediation_status == status)
    if domain:
        domain_obj = db.query(ControlDomain).filter(ControlDomain.code == domain).first()
        if domain_obj:
            q = q.filter(Finding.control_domain_id == domain_obj.id)

    findings = q.order_by(Finding.severity.desc(), Finding.created_at.desc()).all()

    result = []
    for f in findings:
        domain_obj = db.query(ControlDomain).get(f.control_domain_id) if f.control_domain_id else None
        result.append(FindingOut(
            id=f.id,
            assessment_id=f.assessment_id,
            title=f.title,
            description=f.description,
            severity=f.severity,
            likelihood=f.likelihood,
            impact=f.impact,
            control_domain_id=f.control_domain_id,
            control_domain_name=domain_obj.name if domain_obj else None,
            recommendation=f.recommendation,
            owner=f.owner,
            due_date=f.due_date,
            remediation_status=f.remediation_status,
            source_rule=f.source_rule,
            created_at=f.created_at,
        ))
    return result
