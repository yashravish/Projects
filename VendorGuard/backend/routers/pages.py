"""Server-rendered page routes using Jinja2 templates."""

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user, decode_token
from backend.models import (
    User, Vendor, Assessment, Finding, RemediationItem,
    ControlDomain, AssessmentTemplate, AuditLog,
)
from backend.engine.questionnaire import get_questions_for_category, QUESTIONNAIRE_SECTIONS
from backend.services.report_service import generate_html_report

templates = Jinja2Templates(directory="templates")
router = APIRouter(tags=["pages"])


def _try_get_user(request: Request, db: Session):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = decode_token(token)
        username = payload.get("sub")
        if not username:
            return None
        return db.query(User).filter(User.username == username).first()
    except Exception:
        return None


@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})


@router.get("/vendors", response_class=HTMLResponse)
def vendors_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("vendors_list.html", {"request": request, "user": user})


@router.get("/vendors/new", response_class=HTMLResponse)
def vendor_form_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("vendor_form.html", {"request": request, "user": user})


@router.get("/vendors/{vendor_id}", response_class=HTMLResponse)
def vendor_detail_page(request: Request, vendor_id: int, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    vendor = db.query(Vendor).get(vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return templates.TemplateResponse("vendor_detail.html", {
        "request": request, "user": user, "vendor_id": vendor_id,
    })


@router.get("/assessments/{assessment_id}/questionnaire", response_class=HTMLResponse)
def questionnaire_page(request: Request, assessment_id: int, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    vendor = db.query(Vendor).get(assessment.vendor_id)
    sections = get_questions_for_category(vendor.category if vendor else "")
    return templates.TemplateResponse("assessment_questionnaire.html", {
        "request": request, "user": user,
        "assessment_id": assessment_id,
        "vendor": vendor,
        "sections": sections,
    })


@router.get("/assessments/{assessment_id}/results", response_class=HTMLResponse)
def results_page(request: Request, assessment_id: int, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("assessment_results.html", {
        "request": request, "user": user, "assessment_id": assessment_id,
    })


@router.get("/findings", response_class=HTMLResponse)
def findings_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("findings_dashboard.html", {"request": request, "user": user})


@router.get("/remediation", response_class=HTMLResponse)
def remediation_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("remediation_tracker.html", {"request": request, "user": user})


@router.get("/reports/{assessment_id}/preview", response_class=HTMLResponse)
def report_preview_page(request: Request, assessment_id: int, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("report_preview.html", {
        "request": request, "user": user, "assessment_id": assessment_id,
    })


@router.get("/governance", response_class=HTMLResponse)
def governance_page(request: Request, db: Session = Depends(get_db)):
    user = _try_get_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("governance.html", {"request": request, "user": user})
