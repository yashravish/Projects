from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, RemediationItem, Finding, Assessment, Vendor
from backend.schemas import RemediationUpdate, RemediationOut
from backend.services.audit_service import log_action

router = APIRouter(prefix="/api/remediation", tags=["remediation"])


@router.get("", response_model=list[RemediationOut])
def list_remediation(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(RemediationItem)
    if status:
        q = q.filter(RemediationItem.status == status)
    if priority:
        q = q.filter(RemediationItem.priority == priority)

    items = q.order_by(RemediationItem.created_at.desc()).all()
    result = []
    for item in items:
        finding = db.query(Finding).get(item.finding_id)
        vendor_name = ""
        if finding:
            assessment = db.query(Assessment).get(finding.assessment_id)
            if assessment:
                vendor = db.query(Vendor).get(assessment.vendor_id)
                vendor_name = vendor.name if vendor else ""

        result.append(RemediationOut(
            id=item.id,
            finding_id=item.finding_id,
            finding_title=finding.title if finding else "",
            vendor_name=vendor_name,
            action=item.action,
            assigned_to=item.assigned_to,
            priority=item.priority,
            status=item.status,
            due_date=item.due_date,
            completion_date=item.completion_date,
            notes=item.notes,
            created_at=item.created_at,
        ))
    return result


@router.patch("/{item_id}", response_model=RemediationOut)
def update_remediation(
    item_id: int,
    body: RemediationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = db.query(RemediationItem).get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Remediation item not found")

    if body.assigned_to is not None:
        item.assigned_to = body.assigned_to
    if body.status is not None:
        item.status = body.status
        finding = db.query(Finding).get(item.finding_id)
        if finding:
            finding.remediation_status = body.status
        if body.status in ("mitigated", "closed"):
            item.completion_date = date.today()
    if body.due_date is not None:
        item.due_date = body.due_date
        finding = db.query(Finding).get(item.finding_id)
        if finding:
            finding.due_date = body.due_date
    if body.notes is not None:
        item.notes = body.notes

    db.commit()
    db.refresh(item)

    log_action(db, current_user.id, "remediation_updated", "remediation", item_id,
               details=f"Status: {item.status}")

    finding = db.query(Finding).get(item.finding_id)
    vendor_name = ""
    if finding:
        assessment = db.query(Assessment).get(finding.assessment_id)
        if assessment:
            vendor = db.query(Vendor).get(assessment.vendor_id)
            vendor_name = vendor.name if vendor else ""

    return RemediationOut(
        id=item.id,
        finding_id=item.finding_id,
        finding_title=finding.title if finding else "",
        vendor_name=vendor_name,
        action=item.action,
        assigned_to=item.assigned_to,
        priority=item.priority,
        status=item.status,
        due_date=item.due_date,
        completion_date=item.completion_date,
        notes=item.notes,
        created_at=item.created_at,
    )
