"""
Report generation service using Jinja2 templates and WeasyPrint for PDF export.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from backend.models import Assessment, Vendor, Finding, ControlDomain, GeneratedReport

logger = structlog.get_logger()

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "generated"


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_report_data(db: Session, assessment_id: int) -> dict:
    assessment = db.query(Assessment).get(assessment_id)
    if not assessment:
        raise ValueError(f"Assessment {assessment_id} not found")

    vendor = db.query(Vendor).get(assessment.vendor_id)
    findings = (
        db.query(Finding)
        .filter(Finding.assessment_id == assessment_id)
        .order_by(
            Finding.severity.desc(),
            Finding.created_at,
        )
        .all()
    )

    domains = {d.id: d for d in db.query(ControlDomain).all()}

    findings_data = []
    for f in findings:
        domain = domains.get(f.control_domain_id)
        findings_data.append({
            "id": f.id,
            "title": f.title,
            "description": f.description,
            "severity": f.severity,
            "likelihood": f.likelihood,
            "impact": f.impact,
            "domain_name": domain.name if domain else "N/A",
            "domain_code": domain.code if domain else "N/A",
            "recommendation": f.recommendation,
            "remediation_status": f.remediation_status,
            "source_rule": f.source_rule,
        })

    severity_counts = {"Critical": 0, "High": 0, "Moderate": 0, "Low": 0}
    for f in findings_data:
        severity_counts[f["severity"]] = severity_counts.get(f["severity"], 0) + 1

    return {
        "report_date": datetime.now(timezone.utc).strftime("%B %d, %Y"),
        "assessment": {
            "id": assessment.id,
            "type": assessment.assessment_type,
            "phase": assessment.phase.replace("_", " ").title(),
            "status": assessment.status,
            "inherent_risk_score": assessment.inherent_risk_score,
            "inherent_risk_rating": assessment.overall_inherent_risk,
            "residual_risk_score": assessment.residual_risk_score,
            "residual_risk_rating": assessment.overall_residual_risk,
            "executive_summary": assessment.executive_summary or "",
            "ai_summary": assessment.ai_summary or "",
            "created_at": assessment.created_at.strftime("%B %d, %Y") if assessment.created_at else "",
        },
        "vendor": {
            "name": vendor.name,
            "category": vendor.category,
            "description": vendor.description,
            "business_owner": vendor.business_owner,
            "hosting_model": vendor.hosting_model,
            "internet_exposed": vendor.internet_exposed,
            "handles_sensitive_data": vendor.handles_sensitive_data,
            "data_types": vendor.data_types,
            "compliance_attestations": vendor.compliance_attestations,
        },
        "findings": findings_data,
        "severity_counts": severity_counts,
        "total_findings": len(findings_data),
    }


def generate_html_report(db: Session, assessment_id: int) -> str:
    data = _get_report_data(db, assessment_id)
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report_pdf.html")
    return template.render(**data)


def generate_pdf_report(db: Session, assessment_id: int, user_id: int | None = None) -> str:
    _ensure_reports_dir()
    html_content = generate_html_report(db, assessment_id)
    filename = f"assessment_{assessment_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = REPORTS_DIR / filename

    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(str(filepath))
    except ImportError:
        logger.warning("weasyprint_not_available", msg="Saving HTML report instead")
        filepath = filepath.with_suffix(".html")
        filepath.write_text(html_content, encoding="utf-8")

    report_record = GeneratedReport(
        assessment_id=assessment_id,
        report_type="full",
        file_path=str(filepath),
        generated_by=user_id,
    )
    db.add(report_record)
    db.commit()

    logger.info("report_generated", assessment_id=assessment_id, path=str(filepath))
    return str(filepath)
