"""
Deterministic risk rules engine.

Each rule is a function that receives the vendor metadata and a dict of
assessment answers (key -> answer_value).  It returns a finding dict if the
condition is met, or None if the rule does not trigger.

Severity model: Low | Moderate | High | Critical

The engine is intentionally transparent so it can be explained in interviews:
every finding traces back to a named rule and the exact condition that fired.
"""

from typing import Optional


def _answer(answers: dict, key: str, default: str = "") -> str:
    return str(answers.get(key, default)).strip()


def _bool_answer(answers: dict, key: str) -> bool:
    val = _answer(answers, key).lower()
    return val in ("true", "yes", "1")


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def rule_encryption_at_rest(vendor: dict, answers: dict) -> Optional[dict]:
    """Sensitive data without encryption at rest."""
    if vendor.get("handles_sensitive_data") and not _bool_answer(answers, "encryption_rest"):
        data_class = _answer(answers, "data_classification")
        severity = "Critical" if data_class in ("Restricted/Regulated", "Confidential") else "High"
        return {
            "title": "Sensitive data not encrypted at rest",
            "description": (
                f"The vendor handles sensitive data (classification: {data_class or 'unspecified'}) "
                "but does not confirm encryption at rest. This exposes data to unauthorized "
                "access in the event of storage compromise."
            ),
            "severity": severity,
            "likelihood": "High",
            "impact": severity,
            "domain_code": "DP",
            "recommendation": (
                "Require the vendor to implement AES-256 encryption at rest for all "
                "sensitive data stores and provide evidence of key management practices."
            ),
            "source_rule": "rule_encryption_at_rest",
        }
    return None


def rule_encryption_in_transit(vendor: dict, answers: dict) -> Optional[dict]:
    """Internet-exposed service without confirmed TLS."""
    if vendor.get("internet_exposed") and not _bool_answer(answers, "encryption_transit"):
        return {
            "title": "Data in transit not encrypted with TLS 1.2+",
            "description": (
                "The technology is internet-exposed but encryption in transit via TLS 1.2 or "
                "higher has not been confirmed. Data may be intercepted during transmission."
            ),
            "severity": "Critical",
            "likelihood": "High",
            "impact": "Critical",
            "domain_code": "DP",
            "recommendation": (
                "Mandate TLS 1.2+ for all data in transit. Request vendor documentation "
                "of their transport security configuration."
            ),
            "source_rule": "rule_encryption_in_transit",
        }
    return None


def rule_no_mfa_privileged(vendor: dict, answers: dict) -> Optional[dict]:
    """No MFA with privileged access."""
    privileged = _bool_answer(answers, "privileged_access_required")
    mfa = _bool_answer(answers, "mfa_supported")
    if privileged and not mfa:
        return {
            "title": "No MFA available for privileged access",
            "description": (
                "The vendor requires privileged access to the environment but does not "
                "support Multi-Factor Authentication. This significantly increases the risk "
                "of credential compromise leading to unauthorized administrative access."
            ),
            "severity": "High",
            "likelihood": "High",
            "impact": "High",
            "domain_code": "AC",
            "recommendation": (
                "Require the vendor to implement MFA for all administrative and privileged "
                "accounts before granting environment access."
            ),
            "source_rule": "rule_no_mfa_privileged",
        }
    return None


def rule_no_mfa_general(vendor: dict, answers: dict) -> Optional[dict]:
    """No MFA support at all."""
    if not _bool_answer(answers, "mfa_supported") and not _bool_answer(answers, "privileged_access_required"):
        return {
            "title": "Multi-Factor Authentication not supported",
            "description": (
                "The product does not support MFA. Single-factor authentication "
                "increases the risk of unauthorized access through credential theft."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "AC",
            "recommendation": (
                "Request MFA support on the vendor roadmap. Consider compensating controls "
                "such as IP allowlisting or conditional access policies."
            ),
            "source_rule": "rule_no_mfa_general",
        }
    return None


def rule_no_sso(vendor: dict, answers: dict) -> Optional[dict]:
    """No SSO support for enterprise deployment."""
    scope = vendor.get("deployment_scope", "")
    if scope in ("enterprise", "department") and not _bool_answer(answers, "sso_supported"):
        return {
            "title": "SSO not supported for enterprise deployment",
            "description": (
                "The technology is deployed at the enterprise/department level but does not "
                "support Single Sign-On. This creates friction for user management and "
                "increases the risk of orphaned accounts."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "AC",
            "recommendation": (
                "Negotiate SSO integration (SAML 2.0 / OIDC) as a contractual requirement "
                "or identify compensating controls for account lifecycle management."
            ),
            "source_rule": "rule_no_sso",
        }
    return None


def rule_no_audit_logging(vendor: dict, answers: dict) -> Optional[dict]:
    """No audit logging capability."""
    if not _bool_answer(answers, "audit_logging"):
        sensitive = vendor.get("handles_sensitive_data", False)
        severity = "High" if sensitive else "Moderate"
        return {
            "title": "Audit logging not available",
            "description": (
                "The product does not provide audit logging. Without audit trails, "
                "security investigations and compliance evidence collection are significantly impaired."
            ),
            "severity": severity,
            "likelihood": "Moderate",
            "impact": severity,
            "domain_code": "LM",
            "recommendation": (
                "Require the vendor to implement comprehensive audit logging for all "
                "user and administrative actions with timestamps and actor identification."
            ),
            "source_rule": "rule_no_audit_logging",
        }
    return None


def rule_no_log_export(vendor: dict, answers: dict) -> Optional[dict]:
    """Logging exists but cannot be exported."""
    if _bool_answer(answers, "audit_logging") and not _bool_answer(answers, "log_export"):
        return {
            "title": "Audit logs cannot be exported to SIEM",
            "description": (
                "While the product provides audit logging, logs cannot be exported to "
                "an external SIEM or log management platform. This limits centralized "
                "monitoring and incident correlation capabilities."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "LM",
            "recommendation": (
                "Request SIEM integration capability (syslog, API-based export, or "
                "webhook-based log forwarding) from the vendor."
            ),
            "source_rule": "rule_no_log_export",
        }
    return None


def rule_no_incident_response(vendor: dict, answers: dict) -> Optional[dict]:
    """No documented incident response plan."""
    if not _bool_answer(answers, "ir_plan_documented"):
        sensitive = vendor.get("handles_sensitive_data", False)
        severity = "High" if sensitive else "Moderate"
        return {
            "title": "No documented incident response plan",
            "description": (
                "The vendor has not confirmed a documented incident response plan. "
                "Without a formal IR process, breach detection and containment may be "
                "delayed, increasing organizational exposure."
            ),
            "severity": severity,
            "likelihood": "Moderate",
            "impact": severity,
            "domain_code": "IR",
            "recommendation": (
                "Request evidence of the vendor's incident response plan and ensure "
                "contractual obligations include notification timelines and cooperation commitments."
            ),
            "source_rule": "rule_no_incident_response",
        }
    return None


def rule_no_breach_notification(vendor: dict, answers: dict) -> Optional[dict]:
    """No breach notification commitment."""
    if _bool_answer(answers, "ir_plan_documented") and not _bool_answer(answers, "breach_notification_commitment"):
        return {
            "title": "No breach notification timeline commitment",
            "description": (
                "The vendor has an incident response plan but does not commit to a "
                "specific breach notification timeline. Delayed notification may impede "
                "the organization's own incident response obligations."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "IR",
            "recommendation": (
                "Negotiate a contractual breach notification window of 72 hours or less, "
                "aligned with regulatory requirements (e.g., GDPR Article 33)."
            ),
            "source_rule": "rule_no_breach_notification",
        }
    return None


def rule_no_vuln_management(vendor: dict, answers: dict) -> Optional[dict]:
    """No vulnerability management program."""
    if not _bool_answer(answers, "vuln_mgmt_program"):
        return {
            "title": "No formal vulnerability management program",
            "description": (
                "The vendor does not maintain a formal vulnerability management program. "
                "Unpatched vulnerabilities may be exploited, leading to data breaches or service disruption."
            ),
            "severity": "High",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "VU",
            "recommendation": (
                "Require evidence of a vulnerability management program that includes "
                "regular scanning, risk-based prioritization, and defined patching SLAs."
            ),
            "source_rule": "rule_no_vuln_management",
        }
    return None


def rule_slow_patching(vendor: dict, answers: dict) -> Optional[dict]:
    """Slow or undefined patching cadence."""
    cadence = _answer(answers, "patching_cadence")
    if cadence in ("Within 30 days", "No defined cadence", "Unknown"):
        severity = "High" if cadence in ("No defined cadence", "Unknown") else "Moderate"
        return {
            "title": f"Patching cadence concern: {cadence}",
            "description": (
                f"The vendor's critical vulnerability patching cadence is '{cadence}'. "
                "Delayed patching of critical vulnerabilities increases the window of exposure."
            ),
            "severity": severity,
            "likelihood": "Moderate",
            "impact": severity,
            "domain_code": "VU",
            "recommendation": (
                "Negotiate a patching SLA of 7 days or less for critical vulnerabilities. "
                "Request evidence of patch management processes."
            ),
            "source_rule": "rule_slow_patching",
        }
    return None


def rule_no_compliance_evidence(vendor: dict, answers: dict) -> Optional[dict]:
    """No compliance attestations for high-risk vendor."""
    soc2 = _bool_answer(answers, "soc2_certified")
    iso = _bool_answer(answers, "iso27001_certified")
    sensitive = vendor.get("handles_sensitive_data", False)
    if sensitive and not soc2 and not iso:
        return {
            "title": "No compliance attestations for sensitive data handler",
            "description": (
                "The vendor handles sensitive data but holds neither SOC 2 Type II "
                "nor ISO 27001 certification. This reduces assurance that baseline "
                "security controls are independently verified."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "GD",
            "recommendation": (
                "Request SOC 2 Type II or ISO 27001 certification. If unavailable, "
                "conduct an enhanced due diligence review and require compensating controls."
            ),
            "source_rule": "rule_no_compliance_evidence",
        }
    return None


def rule_no_backup_recovery(vendor: dict, answers: dict) -> Optional[dict]:
    """No backup or DR documentation."""
    if not _bool_answer(answers, "backup_procedures") or not _bool_answer(answers, "dr_plan"):
        return {
            "title": "Backup or disaster recovery documentation missing",
            "description": (
                "The vendor has not confirmed documented backup procedures and/or a "
                "disaster recovery plan. Data loss or extended outages may result "
                "from an incident affecting the vendor's infrastructure."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "BC",
            "recommendation": (
                "Request documentation of backup frequency, retention, testing cadence, "
                "and disaster recovery procedures including defined RTO and RPO."
            ),
            "source_rule": "rule_no_backup_recovery",
        }
    return None


def rule_excessive_integrations(vendor: dict, answers: dict) -> Optional[dict]:
    """Many internal system integrations increase attack surface."""
    count = _answer(answers, "integration_count")
    if count in ("4-10", "More than 10"):
        return {
            "title": f"High number of internal integrations ({count})",
            "description": (
                f"The technology integrates with {count} internal systems, expanding the "
                "attack surface and potential blast radius in the event of a compromise."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "SC",
            "recommendation": (
                "Review each integration for least-privilege access. Implement API "
                "gateway controls and monitor integration traffic for anomalies."
            ),
            "source_rule": "rule_excessive_integrations",
        }
    return None


def rule_no_subprocessor_docs(vendor: dict, answers: dict) -> Optional[dict]:
    """No documentation of subprocessors / fourth parties."""
    if not _bool_answer(answers, "subprocessors_documented"):
        return {
            "title": "Subprocessors / fourth parties not documented",
            "description": (
                "The vendor does not maintain or share a list of subprocessors. "
                "Fourth-party risk is opaque, limiting supply chain risk visibility."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "VM",
            "recommendation": (
                "Require the vendor to provide and maintain a current list of subprocessors "
                "with notification of changes."
            ),
            "source_rule": "rule_no_subprocessor_docs",
        }
    return None


def rule_unknown_data_residency(vendor: dict, answers: dict) -> Optional[dict]:
    """Unknown data residency."""
    residency = _answer(answers, "data_residency")
    if residency == "Unknown" and vendor.get("handles_sensitive_data"):
        return {
            "title": "Data residency unknown for sensitive data",
            "description": (
                "The geographic location of data storage is unknown. This may conflict "
                "with regulatory requirements and organizational data sovereignty policies."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "DP",
            "recommendation": (
                "Obtain confirmation of data storage locations and assess alignment "
                "with applicable data residency requirements."
            ),
            "source_rule": "rule_unknown_data_residency",
        }
    return None


# ---------------------------------------------------------------------------
# Category-specific rules
# ---------------------------------------------------------------------------

def rule_ai_broad_data_access(vendor: dict, answers: dict) -> Optional[dict]:
    """AI tool with broad data access and unclear retention."""
    if vendor.get("category") != "AI Tool":
        return None
    scope = _answer(answers, "ai_data_access_scope")
    retention = _answer(answers, "ai_prompt_retention")
    if scope in ("Internal documents", "Email and calendar", "Broad organizational data", "Unknown"):
        severity = "High"
        if retention in ("Retained for improvement", "Retained for model training", "Unknown"):
            severity = "Critical" if scope in ("Broad organizational data",) else "High"
        return {
            "title": "AI tool has broad data access with unclear data handling",
            "description": (
                f"The AI tool accesses '{scope}' and prompt retention is '{retention}'. "
                "Broad access combined with unclear retention policies creates risk of "
                "unintended data exposure and regulatory non-compliance."
            ),
            "severity": severity,
            "likelihood": "High",
            "impact": severity,
            "domain_code": "AG",
            "recommendation": (
                "Restrict AI data access to the minimum necessary scope. Negotiate "
                "a no-retention or session-only retention policy. Disable model "
                "training on customer data."
            ),
            "source_rule": "rule_ai_broad_data_access",
        }
    return None


def rule_ai_training_on_data(vendor: dict, answers: dict) -> Optional[dict]:
    """AI vendor trains models on customer data."""
    if vendor.get("category") != "AI Tool":
        return None
    if _bool_answer(answers, "ai_model_training_on_data"):
        return {
            "title": "Customer data used for AI model training",
            "description": (
                "The vendor uses customer data to train or fine-tune AI models. "
                "This creates risk of sensitive data exposure through model outputs "
                "and may conflict with data processing agreements."
            ),
            "severity": "High",
            "likelihood": "High",
            "impact": "High",
            "domain_code": "AG",
            "recommendation": (
                "Opt out of model training programs. Require contractual guarantees "
                "that customer data is not used for model training."
            ),
            "source_rule": "rule_ai_training_on_data",
        }
    return None


def rule_ai_no_admin_controls(vendor: dict, answers: dict) -> Optional[dict]:
    """AI tool lacks admin controls."""
    if vendor.get("category") != "AI Tool":
        return None
    if not _bool_answer(answers, "ai_admin_controls"):
        return {
            "title": "AI tool lacks administrative access controls",
            "description": (
                "Administrators cannot configure or restrict the AI tool's access scope "
                "and permissions. This limits the organization's ability to enforce "
                "least-privilege principles."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "AG",
            "recommendation": (
                "Require the vendor to provide administrative controls for configuring "
                "AI access scope, user permissions, and data boundaries."
            ),
            "source_rule": "rule_ai_no_admin_controls",
        }
    return None


def rule_iot_no_segmentation(vendor: dict, answers: dict) -> Optional[dict]:
    """IoT platform without network segmentation."""
    if vendor.get("category") != "IoT Platform":
        return None
    if not _bool_answer(answers, "iot_network_segmentation"):
        return {
            "title": "IoT deployment lacks network segmentation",
            "description": (
                "The IoT deployment does not include network segmentation. "
                "Compromised IoT devices on a flat network could provide lateral "
                "movement opportunities to critical systems."
            ),
            "severity": "High",
            "likelihood": "High",
            "impact": "High",
            "domain_code": "IOT",
            "recommendation": (
                "Implement dedicated VLANs or network segments for IoT devices. "
                "Apply firewall rules restricting IoT-to-corporate network communication."
            ),
            "source_rule": "rule_iot_no_segmentation",
        }
    return None


def rule_iot_no_device_inventory(vendor: dict, answers: dict) -> Optional[dict]:
    """IoT platform without device inventory."""
    if vendor.get("category") != "IoT Platform":
        return None
    if not _bool_answer(answers, "iot_device_inventory"):
        return {
            "title": "No automated IoT device inventory",
            "description": (
                "The IoT platform does not provide automated device inventory. "
                "Without knowing what devices are deployed, it is impossible to "
                "ensure all devices are patched, monitored, and accounted for."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "IOT",
            "recommendation": (
                "Require the platform to provide device discovery and inventory features, "
                "or implement a separate asset management solution for IoT devices."
            ),
            "source_rule": "rule_iot_no_device_inventory",
        }
    return None


def rule_iot_default_credentials(vendor: dict, answers: dict) -> Optional[dict]:
    """IoT devices with unchanged default credentials."""
    if vendor.get("category") != "IoT Platform":
        return None
    if not _bool_answer(answers, "iot_default_credentials"):
        return {
            "title": "IoT default credentials not changed on deployment",
            "description": (
                "Factory default credentials are not required to be changed during "
                "deployment. Default credentials are publicly known and trivially "
                "exploited by automated attacks."
            ),
            "severity": "Critical",
            "likelihood": "High",
            "impact": "Critical",
            "domain_code": "IOT",
            "recommendation": (
                "Enforce mandatory credential change during device provisioning. "
                "Implement certificate-based authentication where possible."
            ),
            "source_rule": "rule_iot_default_credentials",
        }
    return None


def rule_token_no_key_management(vendor: dict, answers: dict) -> Optional[dict]:
    """Tokenization without documented key management."""
    if vendor.get("category") != "Tokenization Platform":
        return None
    if not _bool_answer(answers, "token_key_management"):
        return {
            "title": "No documented key management for tokenization platform",
            "description": (
                "The tokenization platform lacks a documented key management process. "
                "Compromised or mismanaged keys could render tokenization ineffective "
                "and expose the underlying sensitive data."
            ),
            "severity": "High",
            "likelihood": "Moderate",
            "impact": "Critical",
            "domain_code": "DP",
            "recommendation": (
                "Require documented key management procedures including key generation, "
                "rotation, storage (preferably HSM-backed), and destruction."
            ),
            "source_rule": "rule_token_no_key_management",
        }
    return None


def rule_token_no_hsm(vendor: dict, answers: dict) -> Optional[dict]:
    """Tokenization without HSM."""
    if vendor.get("category") != "Tokenization Platform":
        return None
    if _bool_answer(answers, "token_key_management") and not _bool_answer(answers, "token_hsm_used"):
        return {
            "title": "Tokenization keys not stored in HSM",
            "description": (
                "Encryption/tokenization keys are not stored in a Hardware Security Module. "
                "Software-based key storage increases the risk of key extraction."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "DP",
            "recommendation": (
                "Evaluate HSM or cloud KMS solutions for cryptographic key storage "
                "to provide hardware-backed key protection."
            ),
            "source_rule": "rule_token_no_hsm",
        }
    return None


def rule_dlt_pii_no_privacy(vendor: dict, answers: dict) -> Optional[dict]:
    """DLT with PII on-chain and no privacy controls."""
    if vendor.get("category") != "Distributed Ledger Platform":
        return None
    if _bool_answer(answers, "dlt_pii_on_chain") and not _bool_answer(answers, "dlt_privacy_controls"):
        return {
            "title": "PII stored on immutable ledger without privacy controls",
            "description": (
                "Personally identifiable information is stored on the distributed ledger "
                "without privacy-preserving controls. Blockchain immutability conflicts "
                "with data subject rights (e.g., right to erasure under GDPR)."
            ),
            "severity": "High",
            "likelihood": "High",
            "impact": "High",
            "domain_code": "DP",
            "recommendation": (
                "Store PII off-chain with only hashed references on the ledger. "
                "Implement privacy-preserving techniques such as zero-knowledge proofs."
            ),
            "source_rule": "rule_dlt_pii_no_privacy",
        }
    return None


def rule_dlt_no_smart_contract_audit(vendor: dict, answers: dict) -> Optional[dict]:
    """Smart contracts not audited."""
    if vendor.get("category") != "Distributed Ledger Platform":
        return None
    audit_status = _answer(answers, "dlt_smart_contract_audit")
    if audit_status in ("No",):
        return {
            "title": "Smart contracts have not been security audited",
            "description": (
                "Smart contracts have not undergone a security audit. Vulnerabilities "
                "in smart contracts can lead to irreversible financial or data loss."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "SC",
            "recommendation": (
                "Commission an independent third-party smart contract security audit "
                "before production deployment."
            ),
            "source_rule": "rule_dlt_no_smart_contract_audit",
        }
    return None


def rule_sw_local_admin_no_update(vendor: dict, answers: dict) -> Optional[dict]:
    """End-user software requiring local admin without auto-update."""
    if vendor.get("category") != "End-User Software Package":
        return None
    admin = _bool_answer(answers, "sw_local_admin_required")
    auto_update = _bool_answer(answers, "sw_auto_update")
    if admin and not auto_update:
        return {
            "title": "Software requires local admin and lacks automatic updates",
            "description": (
                "The software requires elevated local privileges and does not support "
                "automatic updates. This combination increases the risk of running "
                "unpatched software with elevated access."
            ),
            "severity": "High",
            "likelihood": "Moderate",
            "impact": "High",
            "domain_code": "VU",
            "recommendation": (
                "Evaluate whether elevated privileges can be reduced. Implement "
                "a managed software deployment and patching process. Consider "
                "application allowlisting."
            ),
            "source_rule": "rule_sw_local_admin_no_update",
        }
    return None


def rule_sw_not_code_signed(vendor: dict, answers: dict) -> Optional[dict]:
    """End-user software not code-signed."""
    if vendor.get("category") != "End-User Software Package":
        return None
    if not _bool_answer(answers, "sw_code_signed"):
        return {
            "title": "Software is not digitally code-signed",
            "description": (
                "The software is not code-signed by the vendor. Without code signing, "
                "it is difficult to verify the integrity and authenticity of the software."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "SC",
            "recommendation": (
                "Require the vendor to code-sign all distributed binaries. Verify "
                "signatures before enterprise deployment."
            ),
            "source_rule": "rule_sw_not_code_signed",
        }
    return None


def rule_sw_edr_incompatible(vendor: dict, answers: dict) -> Optional[dict]:
    """End-user software incompatible with EDR."""
    if vendor.get("category") != "End-User Software Package":
        return None
    if not _bool_answer(answers, "sw_edr_compatible"):
        return {
            "title": "Software not compatible with endpoint security tools",
            "description": (
                "The software is not confirmed compatible with enterprise EDR and "
                "endpoint security tools. Incompatibility may create monitoring blind spots."
            ),
            "severity": "Moderate",
            "likelihood": "Moderate",
            "impact": "Moderate",
            "domain_code": "LM",
            "recommendation": (
                "Test the software with the organization's EDR solution in a staging "
                "environment. Work with the vendor to resolve any compatibility issues."
            ),
            "source_rule": "rule_sw_edr_incompatible",
        }
    return None


# ---------------------------------------------------------------------------
# Rule registry - all rules in evaluation order
# ---------------------------------------------------------------------------

ALL_RULES = [
    rule_encryption_at_rest,
    rule_encryption_in_transit,
    rule_no_mfa_privileged,
    rule_no_mfa_general,
    rule_no_sso,
    rule_no_audit_logging,
    rule_no_log_export,
    rule_no_incident_response,
    rule_no_breach_notification,
    rule_no_vuln_management,
    rule_slow_patching,
    rule_no_compliance_evidence,
    rule_no_backup_recovery,
    rule_excessive_integrations,
    rule_no_subprocessor_docs,
    rule_unknown_data_residency,
    rule_ai_broad_data_access,
    rule_ai_training_on_data,
    rule_ai_no_admin_controls,
    rule_iot_no_segmentation,
    rule_iot_no_device_inventory,
    rule_iot_default_credentials,
    rule_token_no_key_management,
    rule_token_no_hsm,
    rule_dlt_pii_no_privacy,
    rule_dlt_no_smart_contract_audit,
    rule_sw_local_admin_no_update,
    rule_sw_not_code_signed,
    rule_sw_edr_incompatible,
]


def evaluate_rules(vendor: dict, answers: dict) -> list:
    """Run all rules and return a list of triggered findings."""
    findings = []
    for rule_fn in ALL_RULES:
        result = rule_fn(vendor, answers)
        if result is not None:
            findings.append(result)
    return findings
