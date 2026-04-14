"""
Orchestrator that ties together rules evaluation, scoring, and domain mapping.
"""

from sqlalchemy.orm import Session

from backend.engine.rules import evaluate_rules
from backend.engine.scoring import calculate_inherent_risk, calculate_residual_risk
from backend.models import (
    Vendor, Assessment, AssessmentAnswer, Finding, RemediationItem, ControlDomain,
)


class RiskEngine:
    """
    Stateless engine that evaluates a completed assessment questionnaire
    and produces findings, scores, and remediation items.
    """

    def __init__(self, db: Session):
        self.db = db
        self._domain_cache = {}

    def _get_domain_id(self, code: str) -> int | None:
        if code not in self._domain_cache:
            domain = self.db.query(ControlDomain).filter(ControlDomain.code == code).first()
            self._domain_cache[code] = domain.id if domain else None
        return self._domain_cache[code]

    def evaluate(self, assessment: Assessment) -> dict:
        vendor = self.db.query(Vendor).get(assessment.vendor_id)
        if not vendor:
            raise ValueError(f"Vendor {assessment.vendor_id} not found")

        answers_rows = (
            self.db.query(AssessmentAnswer)
            .filter(AssessmentAnswer.assessment_id == assessment.id)
            .all()
        )
        answers = {a.question_key: a.answer for a in answers_rows}

        vendor_dict = {
            "name": vendor.name,
            "category": vendor.category,
            "handles_sensitive_data": vendor.handles_sensitive_data,
            "internet_exposed": vendor.internet_exposed,
            "deployment_scope": vendor.deployment_scope,
            "hosting_model": vendor.hosting_model,
        }

        raw_findings = evaluate_rules(vendor_dict, answers)

        db_findings = []
        for f in raw_findings:
            domain_id = self._get_domain_id(f["domain_code"])
            finding = Finding(
                assessment_id=assessment.id,
                title=f["title"],
                description=f["description"],
                severity=f["severity"],
                likelihood=f["likelihood"],
                impact=f["impact"],
                control_domain_id=domain_id,
                recommendation=f["recommendation"],
                source_rule=f["source_rule"],
                remediation_status="open",
            )
            self.db.add(finding)
            self.db.flush()

            remediation = RemediationItem(
                finding_id=finding.id,
                action=f["recommendation"],
                priority=f["severity"],
                status="open",
            )
            self.db.add(remediation)
            db_findings.append(finding)

        scoring_input = [
            {
                "severity": f.severity,
                "domain_code": next(
                    (d.code for d in [self.db.query(ControlDomain).get(f.control_domain_id)]
                     if d is not None),
                    "SC",
                ),
                "remediation_status": f.remediation_status,
            }
            for f in db_findings
        ]

        inherent_score, inherent_rating = calculate_inherent_risk(scoring_input)
        residual_score, residual_rating = calculate_residual_risk(inherent_score, scoring_input)

        assessment.inherent_risk_score = inherent_score
        assessment.overall_inherent_risk = inherent_rating
        assessment.residual_risk_score = residual_score
        assessment.overall_residual_risk = residual_rating
        assessment.status = "completed"

        assessment.executive_summary = self._generate_summary(
            vendor, inherent_rating, inherent_score, db_findings
        )

        self.db.commit()

        return {
            "inherent_risk_score": inherent_score,
            "inherent_risk_rating": inherent_rating,
            "residual_risk_score": residual_score,
            "residual_risk_rating": residual_rating,
            "findings_count": len(db_findings),
        }

    @staticmethod
    def _generate_summary(vendor, rating: str, score: float, findings: list) -> str:
        critical = sum(1 for f in findings if f.severity == "Critical")
        high = sum(1 for f in findings if f.severity == "High")
        moderate = sum(1 for f in findings if f.severity == "Moderate")
        low = sum(1 for f in findings if f.severity == "Low")

        lines = [
            f"Security Assessment Summary for {vendor.name}",
            f"",
            f"Overall Inherent Risk: {rating} ({score}/100)",
            f"",
            f"This assessment identified {len(findings)} finding(s) across the "
            f"following severity levels: {critical} Critical, {high} High, "
            f"{moderate} Moderate, and {low} Low.",
            f"",
        ]
        if critical > 0:
            lines.append(
                "Critical findings require immediate attention and should be "
                "addressed before the technology is approved for production use."
            )
        if high > 0:
            lines.append(
                "High-severity findings represent significant risk and should be "
                "remediated within a defined timeline with assigned ownership."
            )
        if rating in ("High", "Critical"):
            lines.append(
                f"\nGiven the {rating.lower()} risk rating, conditional approval "
                "with a mandatory remediation plan is recommended."
            )
        else:
            lines.append(
                f"\nThe {rating.lower()} risk rating suggests acceptable baseline posture, "
                "though identified findings should still be tracked to closure."
            )
        return "\n".join(lines)
