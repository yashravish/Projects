from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, Assessment, GeneratedReport
from backend.services.report_service import generate_html_report, generate_pdf_report
from backend.services.audit_service import log_action

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{assessment_id}", response_class=HTMLResponse)
def get_report_html(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if assessment.status != "completed":
        raise HTTPException(status_code=400, detail="Assessment not yet completed")

    html = generate_html_report(db, assessment_id)
    return HTMLResponse(content=html)


@router.post("/{assessment_id}/generate")
def generate_report(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if assessment.status != "completed":
        raise HTTPException(status_code=400, detail="Assessment not yet completed")

    filepath = generate_pdf_report(db, assessment_id, user_id=current_user.id)
    log_action(db, current_user.id, "report_generated", "report", assessment_id)
    return {"message": "Report generated", "file_path": filepath}


@router.get("/{assessment_id}/download")
def download_report(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    report = (
        db.query(GeneratedReport)
        .filter(GeneratedReport.assessment_id == assessment_id)
        .order_by(GeneratedReport.created_at.desc())
        .first()
    )
    if not report or not report.file_path:
        raise HTTPException(status_code=404, detail="No generated report found")

    import os
    if not os.path.exists(report.file_path):
        raise HTTPException(status_code=404, detail="Report file not found on disk")

    return FileResponse(
        report.file_path,
        filename=os.path.basename(report.file_path),
        media_type="application/pdf" if report.file_path.endswith(".pdf") else "text/html",
    )
