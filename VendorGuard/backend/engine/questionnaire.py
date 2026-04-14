"""
Structured assessment questionnaire definition.

Each section contains questions with a unique key, display text,
answer type (boolean / select / text), and options where applicable.
"""

QUESTIONNAIRE_SECTIONS = {
    "data_handling": {
        "title": "Data Handling & Classification",
        "order": 1,
        "questions": [
            {"key": "data_classification", "text": "What is the highest data classification this technology will handle?", "type": "select", "options": ["Public", "Internal", "Confidential", "Restricted/Regulated"]},
            {"key": "pii_handled", "text": "Will this technology process, store, or transmit Personally Identifiable Information (PII)?", "type": "boolean"},
            {"key": "phi_handled", "text": "Will this technology process, store, or transmit Protected Health Information (PHI)?", "type": "boolean"},
            {"key": "financial_data", "text": "Will this technology handle financial or payment card data?", "type": "boolean"},
            {"key": "data_residency", "text": "Where will data be stored geographically?", "type": "select", "options": ["US Only", "EU Only", "Multiple Regions", "Unknown"]},
            {"key": "data_retention_policy", "text": "Does the vendor have a documented data retention and disposal policy?", "type": "boolean"},
            {"key": "data_deletion_on_termination", "text": "Can data be deleted upon contract termination?", "type": "select", "options": ["Yes, with certification", "Yes, without certification", "No", "Unknown"]},
        ],
    },
    "identity_access": {
        "title": "Identity & Access Management",
        "order": 2,
        "questions": [
            {"key": "sso_supported", "text": "Does the product support Single Sign-On (SSO) integration?", "type": "boolean"},
            {"key": "mfa_supported", "text": "Does the product support Multi-Factor Authentication (MFA)?", "type": "boolean"},
            {"key": "mfa_enforced", "text": "Can MFA be enforced for all users?", "type": "boolean"},
            {"key": "rbac_supported", "text": "Does the product support role-based access control (RBAC)?", "type": "boolean"},
            {"key": "privileged_access_required", "text": "Does the vendor require privileged or admin access to your environment?", "type": "boolean"},
            {"key": "admin_access_scope", "text": "What level of administrative access does the vendor require?", "type": "select", "options": ["None", "Read-only", "Limited write", "Full admin", "Superuser/root"]},
        ],
    },
    "encryption": {
        "title": "Encryption & Data Protection",
        "order": 3,
        "questions": [
            {"key": "encryption_transit", "text": "Is data encrypted in transit using TLS 1.2 or higher?", "type": "boolean"},
            {"key": "encryption_rest", "text": "Is data encrypted at rest?", "type": "boolean"},
            {"key": "encryption_standard", "text": "What encryption standard is used at rest?", "type": "select", "options": ["AES-256", "AES-128", "Other", "Unknown", "Not applicable"]},
            {"key": "key_management", "text": "Who manages encryption keys?", "type": "select", "options": ["Customer managed", "Vendor managed", "Shared responsibility", "Unknown"]},
        ],
    },
    "integrations": {
        "title": "Integrations & API Security",
        "order": 4,
        "questions": [
            {"key": "api_available", "text": "Does the product expose APIs for integration?", "type": "boolean"},
            {"key": "webhook_support", "text": "Does the product use webhooks?", "type": "boolean"},
            {"key": "integration_count", "text": "How many internal systems will this technology integrate with?", "type": "select", "options": ["0", "1-3", "4-10", "More than 10"]},
            {"key": "api_authentication", "text": "How are API calls authenticated?", "type": "select", "options": ["OAuth 2.0", "API Key", "Basic Auth", "Certificate", "None", "Unknown"]},
        ],
    },
    "logging_monitoring": {
        "title": "Logging & Monitoring",
        "order": 5,
        "questions": [
            {"key": "audit_logging", "text": "Does the product provide audit logging of user and admin actions?", "type": "boolean"},
            {"key": "log_export", "text": "Can audit logs be exported to an external SIEM or log management platform?", "type": "boolean"},
            {"key": "monitoring_available", "text": "Does the vendor provide uptime and availability monitoring?", "type": "boolean"},
            {"key": "log_retention_period", "text": "What is the audit log retention period?", "type": "select", "options": ["Less than 30 days", "30-90 days", "90-365 days", "More than 365 days", "Unknown"]},
        ],
    },
    "incident_response": {
        "title": "Incident Response",
        "order": 6,
        "questions": [
            {"key": "ir_plan_documented", "text": "Does the vendor have a documented incident response plan?", "type": "boolean"},
            {"key": "breach_notification_commitment", "text": "Does the vendor commit to breach notification timelines?", "type": "boolean"},
            {"key": "breach_notification_hours", "text": "What is the committed breach notification timeline?", "type": "select", "options": ["24 hours", "48 hours", "72 hours", "No commitment", "Unknown"]},
            {"key": "security_contact_available", "text": "Does the vendor provide a dedicated security point of contact?", "type": "boolean"},
        ],
    },
    "vulnerability_mgmt": {
        "title": "Vulnerability Management",
        "order": 7,
        "questions": [
            {"key": "vuln_mgmt_program", "text": "Does the vendor have a formal vulnerability management program?", "type": "boolean"},
            {"key": "patching_cadence", "text": "What is the vendor's patching cadence for critical vulnerabilities?", "type": "select", "options": ["Within 24 hours", "Within 7 days", "Within 30 days", "No defined cadence", "Unknown"]},
            {"key": "pentest_frequency", "text": "Does the vendor conduct regular penetration testing?", "type": "select", "options": ["Annually", "Semi-annually", "Quarterly", "Never", "Unknown"]},
        ],
    },
    "compliance": {
        "title": "Compliance & Governance",
        "order": 8,
        "questions": [
            {"key": "soc2_certified", "text": "Does the vendor hold SOC 2 Type II certification?", "type": "boolean"},
            {"key": "iso27001_certified", "text": "Does the vendor hold ISO 27001 certification?", "type": "boolean"},
            {"key": "gdpr_compliant", "text": "Is the vendor GDPR compliant?", "type": "select", "options": ["Yes, documented", "Claims compliance", "Not applicable", "Unknown"]},
            {"key": "hipaa_compliant", "text": "Is the vendor HIPAA compliant (if applicable)?", "type": "select", "options": ["Yes, with BAA", "Claims compliance", "Not applicable", "Unknown"]},
            {"key": "subprocessors_documented", "text": "Does the vendor maintain and publish a list of subprocessors / fourth parties?", "type": "boolean"},
            {"key": "right_to_audit", "text": "Does the contract include right-to-audit clauses?", "type": "boolean"},
        ],
    },
    "business_continuity": {
        "title": "Business Continuity & Disaster Recovery",
        "order": 9,
        "questions": [
            {"key": "backup_procedures", "text": "Does the vendor have documented backup procedures?", "type": "boolean"},
            {"key": "dr_plan", "text": "Does the vendor have a documented disaster recovery plan?", "type": "boolean"},
            {"key": "rto_rpo_defined", "text": "Are Recovery Time Objective (RTO) and Recovery Point Objective (RPO) defined?", "type": "boolean"},
            {"key": "sla_uptime", "text": "What is the committed SLA uptime?", "type": "select", "options": ["99.99%", "99.9%", "99.5%", "99%", "No SLA", "Unknown"]},
        ],
    },
    "ai_specific": {
        "title": "AI-Specific Controls",
        "order": 10,
        "category_filter": "AI Tool",
        "questions": [
            {"key": "ai_data_access_scope", "text": "What organizational data does the AI tool have access to?", "type": "select", "options": ["User-provided input only", "Internal documents", "Email and calendar", "Broad organizational data", "Unknown"]},
            {"key": "ai_prompt_retention", "text": "Does the AI vendor retain user prompts or conversation data?", "type": "select", "options": ["No retention", "Temporary (session only)", "Retained for improvement", "Retained for model training", "Unknown"]},
            {"key": "ai_model_training_on_data", "text": "Is customer data used to train or fine-tune the AI model?", "type": "boolean"},
            {"key": "ai_output_controls", "text": "Are there controls to prevent sensitive data leakage in AI outputs?", "type": "boolean"},
            {"key": "ai_admin_controls", "text": "Can administrators configure and restrict AI access scope and permissions?", "type": "boolean"},
        ],
    },
    "iot_specific": {
        "title": "IoT-Specific Controls",
        "order": 11,
        "category_filter": "IoT Platform",
        "questions": [
            {"key": "iot_device_inventory", "text": "Does the platform maintain an automated device inventory?", "type": "boolean"},
            {"key": "iot_network_segmentation", "text": "Does the deployment include network segmentation for IoT devices?", "type": "boolean"},
            {"key": "iot_firmware_updates", "text": "Does the vendor support secure, authenticated firmware updates?", "type": "boolean"},
            {"key": "iot_default_credentials", "text": "Are factory default credentials required to be changed during deployment?", "type": "boolean"},
            {"key": "iot_physical_security", "text": "Are IoT devices physically secured against unauthorized tampering?", "type": "select", "options": ["Yes", "Partially", "No", "Not applicable"]},
        ],
    },
    "tokenization_specific": {
        "title": "Tokenization-Specific Controls",
        "order": 12,
        "category_filter": "Tokenization Platform",
        "questions": [
            {"key": "token_key_management", "text": "Is there a documented cryptographic key management process?", "type": "boolean"},
            {"key": "token_hsm_used", "text": "Are Hardware Security Modules (HSMs) used for key storage?", "type": "boolean"},
            {"key": "token_pci_scope", "text": "Does the tokenization solution reduce PCI DSS scope?", "type": "select", "options": ["Yes, documented", "Partially", "No", "Unknown"]},
            {"key": "token_detokenization_controls", "text": "Are detokenization access controls and restrictions documented?", "type": "boolean"},
        ],
    },
    "dlt_specific": {
        "title": "Distributed Ledger / Blockchain Controls",
        "order": 13,
        "category_filter": "Distributed Ledger Platform",
        "questions": [
            {"key": "dlt_pii_on_chain", "text": "Does the ledger contain any PII or sensitive data on-chain?", "type": "boolean"},
            {"key": "dlt_privacy_controls", "text": "Are privacy-preserving controls implemented (e.g., zero-knowledge proofs, private channels)?", "type": "boolean"},
            {"key": "dlt_node_security", "text": "Are DLT nodes hardened and access-controlled?", "type": "boolean"},
            {"key": "dlt_smart_contract_audit", "text": "Have smart contracts been independently security audited?", "type": "select", "options": ["Yes, by third party", "Yes, internal only", "No", "Not applicable"]},
            {"key": "dlt_consensus_documented", "text": "Is the consensus mechanism documented and assessed for the use case?", "type": "boolean"},
        ],
    },
    "enduser_software": {
        "title": "End-User Software Controls",
        "order": 14,
        "category_filter": "End-User Software Package",
        "questions": [
            {"key": "sw_local_admin_required", "text": "Does the software require local admin or elevated privileges to install or run?", "type": "boolean"},
            {"key": "sw_auto_update", "text": "Does the software support automatic or centrally managed updates?", "type": "boolean"},
            {"key": "sw_network_access", "text": "Does the software require outbound network or internet access?", "type": "boolean"},
            {"key": "sw_local_data_storage", "text": "Does the software store data locally on the endpoint?", "type": "boolean"},
            {"key": "sw_code_signed", "text": "Is the software digitally code-signed by the vendor?", "type": "boolean"},
            {"key": "sw_edr_compatible", "text": "Is the software compatible with enterprise EDR and endpoint security tools?", "type": "boolean"},
        ],
    },
}


def get_questions_for_category(category: str) -> dict:
    """Return only the questionnaire sections applicable to the given vendor category."""
    result = {}
    for key, section in QUESTIONNAIRE_SECTIONS.items():
        cat_filter = section.get("category_filter")
        if cat_filter is None or cat_filter == category:
            result[key] = section
    return result


def get_all_question_keys() -> list:
    """Return a flat list of all question keys."""
    keys = []
    for section in QUESTIONNAIRE_SECTIONS.values():
        for q in section["questions"]:
            keys.append(q["key"])
    return keys
