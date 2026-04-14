from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, Vendor, Assessment, AssessmentAnswer
from backend.schemas import AssessmentCreate, AssessmentSubmit, AssessmentOut
from backend.engine.risk_engine import RiskEngine
from backend.services.audit_service import log_action
from backend.services.ai_service import generate_executive_summary, is_ai_enabled

router = APIRouter(prefix="/api/assessments", tags=["assessments"])


@router.get("", response_model=list[AssessmentOut])
def list_assessments(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    assessments = db.query(Assessment).order_by(Assessment.created_at.desc()).all()
    result = []
    for a in assessments:
        vendor = db.query(Vendor).get(a.vendor_id)
        out = AssessmentOut(
            id=a.id,
            vendor_id=a.vendor_id,
            vendor_name=vendor.name if vendor else "",
            assessment_type=a.assessment_type,
            phase=a.phase,
            assessor_id=a.assessor_id,
            overall_inherent_risk=a.overall_inherent_risk,
            inherent_risk_score=a.inherent_risk_score,
            overall_residual_risk=a.overall_residual_risk,
            residual_risk_score=a.residual_risk_score,
            status=a.status,
            executive_summary=a.executive_summary,
            ai_summary=a.ai_summary,
            findings_count=len(a.findings),
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        result.append(out)
    return result


@router.post("", response_model=AssessmentOut, status_code=201)
def create_assessment(
    body: AssessmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    vendor = db.query(Vendor).get(body.vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")

    assessment = Assessment(
        vendor_id=body.vendor_id,
        assessment_type=body.assessment_type,
        phase=body.phase,
        assessor_id=current_user.id,
        status="draft",
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    log_action(db, current_user.id, "assessment_created", "assessment", assessment.id,
               details=f"Vendor: {vendor.name}")
    return AssessmentOut(
        id=assessment.id,
        vendor_id=assessment.vendor_id,
        vendor_name=vendor.name,
        assessment_type=assessment.assessment_type,
        phase=assessment.phase,
        assessor_id=assessment.assessor_id,
        status=assessment.status,
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
    )


@router.get("/{assessment_id}", response_model=AssessmentOut)
def get_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    vendor = db.query(Vendor).get(assessment.vendor_id)
    return AssessmentOut(
        id=assessment.id,
        vendor_id=assessment.vendor_id,
        vendor_name=vendor.name if vendor else "",
        assessment_type=assessment.assessment_type,
        phase=assessment.phase,
        assessor_id=assessment.assessor_id,
        overall_inherent_risk=assessment.overall_inherent_risk,
        inherent_risk_score=assessment.inherent_risk_score,
        overall_residual_risk=assessment.overall_residual_risk,
        residual_risk_score=assessment.residual_risk_score,
        status=assessment.status,
        executive_summary=assessment.executive_summary,
        ai_summary=assessment.ai_summary,
        findings_count=len(assessment.findings),
        created_at=assessment.created_at,
        updated_at=assessment.updated_at,
    )


@router.post("/{assessment_id}/submit")
def submit_answers(
    assessment_id: int,
    body: AssessmentSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    db.query(AssessmentAnswer).filter(AssessmentAnswer.assessment_id == assessment_id).delete()

    for ans in body.answers:
        db.add(AssessmentAnswer(
            assessment_id=assessment_id,
            question_key=ans.question_key,
            section=ans.section,
            question_text=ans.question_text,
            answer=ans.answer,
            notes=ans.notes,
        ))
    assessment.status = "in_progress"
    db.commit()

    log_action(db, current_user.id, "assessment_submitted", "assessment", assessment_id)
    return {"message": "Answers submitted", "assessment_id": assessment_id}


@router.post("/{assessment_id}/evaluate")
def evaluate_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    engine = RiskEngine(db)
    result = engine.evaluate(assessment)

    if is_ai_enabled():
        vendor = db.query(Vendor).get(assessment.vendor_id)
        findings_summary = [
            {"severity": f.severity, "title": f.title}
            for f in assessment.findings
        ]
        ai_text = generate_executive_summary(
            vendor.name, vendor.category, findings_summary,
            result["inherent_risk_rating"], result["inherent_risk_score"],
        )
        if ai_text:
            assessment.ai_summary = ai_text
            db.commit()

    log_action(db, current_user.id, "assessment_evaluated", "assessment", assessment_id,
               details=f"Risk: {result['inherent_risk_rating']}")
    return result
