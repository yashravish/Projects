"""
Database seeding script.

Creates demo users, control domains, vendors, sample assessments with
findings, and assessment templates.

Usage: python -m backend.seed
"""

import json
import sys
from datetime import date, timedelta

from backend.database import SessionLocal, engine, Base
from backend.auth import get_password_hash
from backend.models import (
    User, Vendor, VendorIntegration, ControlDomain, Assessment,
    AssessmentAnswer, Finding, RemediationItem, AssessmentTemplate, AuditLog,
)
from backend.engine.domain_mapping import CONTROL_DOMAINS
from backend.engine.questionnaire import QUESTIONNAIRE_SECTIONS
from backend.engine.risk_engine import RiskEngine


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    if db.query(User).first():
        print("Database already seeded. Skipping.")
        db.close()
        return

    print("Seeding database...")

    # --- Users ---
    admin = User(
        username="admin", email="admin@vendorguard.local",
        hashed_password=get_password_hash("admin123"),
        full_name="Security Admin", role="admin",
    )
    analyst = User(
        username="analyst", email="analyst@vendorguard.local",
        hashed_password=get_password_hash("analyst123"),
        full_name="Jane Doe", role="analyst",
    )
    db.add_all([admin, analyst])
    db.flush()

    # --- Control Domains ---
    for d in CONTROL_DOMAINS:
        db.add(ControlDomain(**d))
    db.flush()

    # --- Vendors ---
    vendors_data = [
        {
            "name": "PeopleForce HR",
            "category": "SaaS",
            "description": "Cloud-based HR management platform for employee records, benefits, and payroll processing.",
            "website": "https://peopleforce.example.com",
            "business_owner": "VP of Human Resources",
            "vendor_contact": "security@peopleforce.example.com",
            "hosting_model": "cloud",
            "deployment_scope": "enterprise",
            "internet_exposed": True,
            "handles_sensitive_data": True,
            "data_types": ["PII", "Financial", "Employee Records"],
            "compliance_attestations": ["SOC 2 Type II"],
            "status": "active",
        },
        {
            "name": "NoteGenius AI",
            "category": "AI Tool",
            "description": "AI-powered meeting note-taker and document summarizer integrated with corporate document stores.",
            "website": "https://notegenius.example.com",
            "business_owner": "Director of Product",
            "vendor_contact": "trust@notegenius.example.com",
            "hosting_model": "cloud",
            "deployment_scope": "enterprise",
            "internet_exposed": True,
            "handles_sensitive_data": True,
            "data_types": ["Internal Documents", "Meeting Notes", "PII"],
            "compliance_attestations": [],
            "status": "active",
        },
        {
            "name": "OfficeSense IoT",
            "category": "IoT Platform",
            "description": "Smart badge sensors and occupancy monitoring for office buildings.",
            "website": "https://officesense.example.com",
            "business_owner": "Facilities Manager",
            "vendor_contact": "support@officesense.example.com",
            "hosting_model": "hybrid",
            "deployment_scope": "enterprise",
            "internet_exposed": True,
            "handles_sensitive_data": True,
            "data_types": ["PII", "Location Data", "Badge Scans"],
            "compliance_attestations": [],
            "status": "active",
        },
        {
            "name": "VaultPay Tokenization",
            "category": "Tokenization Platform",
            "description": "Payment tokenization service for PCI scope reduction in e-commerce transactions.",
            "website": "https://vaultpay.example.com",
            "business_owner": "VP of Engineering",
            "vendor_contact": "compliance@vaultpay.example.com",
            "hosting_model": "cloud",
            "deployment_scope": "enterprise",
            "internet_exposed": True,
            "handles_sensitive_data": True,
            "data_types": ["Payment Card Data", "Financial", "PII"],
            "compliance_attestations": ["PCI DSS Level 1", "SOC 2 Type II"],
            "status": "active",
        },
        {
            "name": "AuditChain DLT",
            "category": "Distributed Ledger Platform",
            "description": "Distributed ledger platform for immutable audit trail records and compliance evidence.",
            "website": "https://auditchain.example.com",
            "business_owner": "Chief Compliance Officer",
            "vendor_contact": "security@auditchain.example.com",
            "hosting_model": "hybrid",
            "deployment_scope": "department",
            "internet_exposed": False,
            "handles_sensitive_data": True,
            "data_types": ["Audit Records", "Compliance Evidence", "PII"],
            "compliance_attestations": ["ISO 27001"],
            "status": "active",
        },
    ]

    vendor_objs = []
    for v_data in vendors_data:
        dt = v_data.pop("data_types")
        ca = v_data.pop("compliance_attestations")
        v = Vendor(**v_data, data_types_json=json.dumps(dt), compliance_attestations_json=json.dumps(ca), created_by=admin.id)
        db.add(v)
        db.flush()
        vendor_objs.append(v)

    # --- Integrations ---
    integrations = [
        (vendor_objs[0], "Active Directory", "SSO", "inbound", "SAML SSO for employee authentication"),
        (vendor_objs[0], "Payroll System", "API", "outbound", "Payroll data sync"),
        (vendor_objs[1], "SharePoint", "API", "inbound", "Document access for AI summarization"),
        (vendor_objs[1], "Microsoft Teams", "webhook", "bidirectional", "Meeting integration"),
        (vendor_objs[2], "Building Management System", "API", "bidirectional", "Occupancy data exchange"),
        (vendor_objs[3], "E-Commerce Platform", "API", "bidirectional", "Payment processing"),
        (vendor_objs[3], "Payment Gateway", "API", "outbound", "Token exchange"),
        (vendor_objs[4], "ERP System", "API", "inbound", "Audit record ingestion"),
    ]
    for vendor, sys_name, int_type, direction, desc in integrations:
        db.add(VendorIntegration(
            vendor_id=vendor.id, system_name=sys_name,
            integration_type=int_type, data_flow_direction=direction, description=desc,
        ))
    db.flush()

    # --- Assessments with sample answers ---
    assessment_answers_map = {
        0: {  # PeopleForce HR (SaaS)
            "data_classification": "Restricted/Regulated", "pii_handled": "true",
            "phi_handled": "false", "financial_data": "true",
            "data_residency": "US Only", "data_retention_policy": "true",
            "data_deletion_on_termination": "Yes, with certification",
            "sso_supported": "true", "mfa_supported": "true", "mfa_enforced": "true",
            "rbac_supported": "true", "privileged_access_required": "false",
            "admin_access_scope": "None",
            "encryption_transit": "true", "encryption_rest": "true",
            "encryption_standard": "AES-256", "key_management": "Vendor managed",
            "api_available": "true", "webhook_support": "false",
            "integration_count": "1-3", "api_authentication": "OAuth 2.0",
            "audit_logging": "true", "log_export": "false",
            "monitoring_available": "true", "log_retention_period": "90-365 days",
            "ir_plan_documented": "true", "breach_notification_commitment": "true",
            "breach_notification_hours": "72 hours", "security_contact_available": "true",
            "vuln_mgmt_program": "true", "patching_cadence": "Within 7 days",
            "pentest_frequency": "Annually",
            "soc2_certified": "true", "iso27001_certified": "false",
            "gdpr_compliant": "Yes, documented", "hipaa_compliant": "Not applicable",
            "subprocessors_documented": "true", "right_to_audit": "false",
            "backup_procedures": "true", "dr_plan": "false",
            "rto_rpo_defined": "false", "sla_uptime": "99.9%",
        },
        1: {  # NoteGenius AI
            "data_classification": "Confidential", "pii_handled": "true",
            "phi_handled": "false", "financial_data": "false",
            "data_residency": "Multiple Regions", "data_retention_policy": "false",
            "data_deletion_on_termination": "Unknown",
            "sso_supported": "false", "mfa_supported": "false", "mfa_enforced": "false",
            "rbac_supported": "false", "privileged_access_required": "true",
            "admin_access_scope": "Full admin",
            "encryption_transit": "true", "encryption_rest": "false",
            "encryption_standard": "Unknown", "key_management": "Unknown",
            "api_available": "true", "webhook_support": "true",
            "integration_count": "4-10", "api_authentication": "API Key",
            "audit_logging": "false", "log_export": "false",
            "monitoring_available": "false", "log_retention_period": "Unknown",
            "ir_plan_documented": "false", "breach_notification_commitment": "false",
            "breach_notification_hours": "No commitment", "security_contact_available": "false",
            "vuln_mgmt_program": "false", "patching_cadence": "Unknown",
            "pentest_frequency": "Unknown",
            "soc2_certified": "false", "iso27001_certified": "false",
            "gdpr_compliant": "Unknown", "hipaa_compliant": "Not applicable",
            "subprocessors_documented": "false", "right_to_audit": "false",
            "backup_procedures": "false", "dr_plan": "false",
            "rto_rpo_defined": "false", "sla_uptime": "No SLA",
            "ai_data_access_scope": "Broad organizational data",
            "ai_prompt_retention": "Retained for model training",
            "ai_model_training_on_data": "true",
            "ai_output_controls": "false", "ai_admin_controls": "false",
        },
        2: {  # OfficeSense IoT
            "data_classification": "Confidential", "pii_handled": "true",
            "phi_handled": "false", "financial_data": "false",
            "data_residency": "US Only", "data_retention_policy": "true",
            "data_deletion_on_termination": "Yes, without certification",
            "sso_supported": "false", "mfa_supported": "true", "mfa_enforced": "false",
            "rbac_supported": "true", "privileged_access_required": "true",
            "admin_access_scope": "Limited write",
            "encryption_transit": "true", "encryption_rest": "true",
            "encryption_standard": "AES-128", "key_management": "Vendor managed",
            "api_available": "true", "webhook_support": "true",
            "integration_count": "1-3", "api_authentication": "API Key",
            "audit_logging": "true", "log_export": "true",
            "monitoring_available": "true", "log_retention_period": "30-90 days",
            "ir_plan_documented": "true", "breach_notification_commitment": "false",
            "breach_notification_hours": "No commitment", "security_contact_available": "true",
            "vuln_mgmt_program": "true", "patching_cadence": "Within 30 days",
            "pentest_frequency": "Annually",
            "soc2_certified": "false", "iso27001_certified": "false",
            "gdpr_compliant": "Claims compliance", "hipaa_compliant": "Not applicable",
            "subprocessors_documented": "false", "right_to_audit": "false",
            "backup_procedures": "true", "dr_plan": "true",
            "rto_rpo_defined": "true", "sla_uptime": "99.5%",
            "iot_device_inventory": "false", "iot_network_segmentation": "false",
            "iot_firmware_updates": "true", "iot_default_credentials": "false",
            "iot_physical_security": "Partially",
        },
        3: {  # VaultPay Tokenization
            "data_classification": "Restricted/Regulated", "pii_handled": "true",
            "phi_handled": "false", "financial_data": "true",
            "data_residency": "US Only", "data_retention_policy": "true",
            "data_deletion_on_termination": "Yes, with certification",
            "sso_supported": "true", "mfa_supported": "true", "mfa_enforced": "true",
            "rbac_supported": "true", "privileged_access_required": "false",
            "admin_access_scope": "None",
            "encryption_transit": "true", "encryption_rest": "true",
            "encryption_standard": "AES-256", "key_management": "Shared responsibility",
            "api_available": "true", "webhook_support": "true",
            "integration_count": "4-10", "api_authentication": "OAuth 2.0",
            "audit_logging": "true", "log_export": "true",
            "monitoring_available": "true", "log_retention_period": "More than 365 days",
            "ir_plan_documented": "true", "breach_notification_commitment": "true",
            "breach_notification_hours": "24 hours", "security_contact_available": "true",
            "vuln_mgmt_program": "true", "patching_cadence": "Within 24 hours",
            "pentest_frequency": "Quarterly",
            "soc2_certified": "true", "iso27001_certified": "true",
            "gdpr_compliant": "Yes, documented", "hipaa_compliant": "Not applicable",
            "subprocessors_documented": "true", "right_to_audit": "true",
            "backup_procedures": "true", "dr_plan": "true",
            "rto_rpo_defined": "true", "sla_uptime": "99.99%",
            "token_key_management": "false", "token_hsm_used": "false",
            "token_pci_scope": "Partially", "token_detokenization_controls": "true",
        },
        4: {  # AuditChain DLT
            "data_classification": "Confidential", "pii_handled": "true",
            "phi_handled": "false", "financial_data": "false",
            "data_residency": "US Only", "data_retention_policy": "true",
            "data_deletion_on_termination": "No",
            "sso_supported": "true", "mfa_supported": "true", "mfa_enforced": "true",
            "rbac_supported": "true", "privileged_access_required": "false",
            "admin_access_scope": "Read-only",
            "encryption_transit": "true", "encryption_rest": "true",
            "encryption_standard": "AES-256", "key_management": "Customer managed",
            "api_available": "true", "webhook_support": "false",
            "integration_count": "1-3", "api_authentication": "Certificate",
            "audit_logging": "true", "log_export": "true",
            "monitoring_available": "true", "log_retention_period": "More than 365 days",
            "ir_plan_documented": "true", "breach_notification_commitment": "true",
            "breach_notification_hours": "48 hours", "security_contact_available": "true",
            "vuln_mgmt_program": "true", "patching_cadence": "Within 7 days",
            "pentest_frequency": "Semi-annually",
            "soc2_certified": "true", "iso27001_certified": "true",
            "gdpr_compliant": "Yes, documented", "hipaa_compliant": "Not applicable",
            "subprocessors_documented": "true", "right_to_audit": "true",
            "backup_procedures": "true", "dr_plan": "true",
            "rto_rpo_defined": "true", "sla_uptime": "99.9%",
            "dlt_pii_on_chain": "true", "dlt_privacy_controls": "false",
            "dlt_node_security": "true",
            "dlt_smart_contract_audit": "Yes, internal only",
            "dlt_consensus_documented": "true",
        },
    }

    for idx, vendor in enumerate(vendor_objs):
        assessment = Assessment(
            vendor_id=vendor.id,
            assessment_type="initial",
            phase="pre_implementation",
            assessor_id=analyst.id,
            status="in_progress",
        )
        db.add(assessment)
        db.flush()

        answers = assessment_answers_map.get(idx, {})
        q_key_to_text = {}
        for section in QUESTIONNAIRE_SECTIONS.values():
            for q in section["questions"]:
                q_key_to_text[q["key"]] = q["text"]

        for key, value in answers.items():
            section_name = ""
            for sec_key, sec_data in QUESTIONNAIRE_SECTIONS.items():
                if any(q["key"] == key for q in sec_data["questions"]):
                    section_name = sec_key
                    break
            db.add(AssessmentAnswer(
                assessment_id=assessment.id,
                question_key=key,
                section=section_name,
                question_text=q_key_to_text.get(key, ""),
                answer=value,
            ))
        db.flush()

        engine = RiskEngine(db)
        engine.evaluate(assessment)

        # Set some remediation due dates and statuses for realism
        for finding in assessment.findings:
            for rem in finding.remediation_items:
                rem.due_date = date.today() + timedelta(days=30 * (1 if finding.severity == "Critical" else 2 if finding.severity == "High" else 3))
                rem.assigned_to = "Jane Doe" if idx % 2 == 0 else "Security Team"
                if idx == 0 and finding.severity == "Moderate":
                    rem.status = "in_progress"
                    finding.remediation_status = "in_progress"
        db.flush()

    # --- Assessment Templates ---
    for sec_key, sec_data in QUESTIONNAIRE_SECTIONS.items():
        cat_filter = sec_data.get("category_filter", "All Categories")
        db.add(AssessmentTemplate(
            name=f"{sec_data['title']} Template",
            category=cat_filter,
            description=f"Standard assessment questions for {sec_data['title'].lower()}.",
            questions_json=json.dumps(sec_data["questions"]),
            is_active=True,
        ))

    db.commit()
    db.close()
    print("Database seeded successfully!")
    print("Demo credentials:")
    print("  Admin:   admin / admin123")
    print("  Analyst: analyst / analyst123")


if __name__ == "__main__":
    seed()
