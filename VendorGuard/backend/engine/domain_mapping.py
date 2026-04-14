"""
Control domain definitions inspired by NIST CSF and ISO 27001.

Disclaimer: These mappings are illustrative and educational. They are not
formal compliance advice and do not represent a complete implementation of
NIST or ISO 27001 controls.
"""

CONTROL_DOMAINS = [
    {
        "code": "AC",
        "name": "Access Control",
        "description": "Controls related to identity management, authentication, and authorization.",
        "nist_mapping": "PR.AC (Protect: Access Control)",
        "iso_mapping": "A.9 Access Control",
    },
    {
        "code": "DP",
        "name": "Data Protection",
        "description": "Controls for data classification, encryption, privacy, and data lifecycle management.",
        "nist_mapping": "PR.DS (Protect: Data Security)",
        "iso_mapping": "A.8 Asset Management / A.10 Cryptography",
    },
    {
        "code": "AM",
        "name": "Asset Management",
        "description": "Controls for maintaining inventory and ownership of information assets.",
        "nist_mapping": "ID.AM (Identify: Asset Management)",
        "iso_mapping": "A.8 Asset Management",
    },
    {
        "code": "VM",
        "name": "Vendor Management",
        "description": "Controls for supply chain risk, third-party oversight, and contractual security.",
        "nist_mapping": "ID.SC (Identify: Supply Chain Risk Management)",
        "iso_mapping": "A.15 Supplier Relationships",
    },
    {
        "code": "LM",
        "name": "Logging and Monitoring",
        "description": "Controls for audit trails, security monitoring, and event detection.",
        "nist_mapping": "DE.CM (Detect: Continuous Monitoring)",
        "iso_mapping": "A.12.4 Logging and Monitoring",
    },
    {
        "code": "IR",
        "name": "Incident Response",
        "description": "Controls for incident detection, reporting, response, and recovery.",
        "nist_mapping": "RS (Respond) / RC (Recover)",
        "iso_mapping": "A.16 Information Security Incident Management",
    },
    {
        "code": "VU",
        "name": "Vulnerability Management",
        "description": "Controls for vulnerability identification, patching, and remediation.",
        "nist_mapping": "DE.CM / RS.MI (Detect / Respond: Mitigation)",
        "iso_mapping": "A.12.6 Technical Vulnerability Management",
    },
    {
        "code": "SC",
        "name": "Secure Configuration",
        "description": "Controls for baseline security configurations and hardening.",
        "nist_mapping": "PR.IP (Protect: Information Protection)",
        "iso_mapping": "A.14 System Acquisition, Development and Maintenance",
    },
    {
        "code": "BC",
        "name": "Business Continuity",
        "description": "Controls for backup, disaster recovery, and service continuity.",
        "nist_mapping": "PR.IP / RC.RP (Recover: Recovery Planning)",
        "iso_mapping": "A.17 Business Continuity",
    },
    {
        "code": "GD",
        "name": "Governance and Documentation",
        "description": "Controls for policies, procedures, compliance documentation, and security governance.",
        "nist_mapping": "ID.GV (Identify: Governance)",
        "iso_mapping": "A.5 Information Security Policies / A.18 Compliance",
    },
    {
        "code": "AG",
        "name": "AI Governance",
        "description": "Controls specific to AI/ML systems: data usage, model transparency, and output controls.",
        "nist_mapping": "NIST AI RMF (AI Risk Management Framework)",
        "iso_mapping": "ISO/IEC 42001 AI Management System (emerging)",
    },
    {
        "code": "IOT",
        "name": "IoT Security",
        "description": "Controls specific to IoT devices: segmentation, firmware, physical security, and device management.",
        "nist_mapping": "NISTIR 8259 (IoT Cybersecurity)",
        "iso_mapping": "ISO/IEC 27400 IoT Security and Privacy (emerging)",
    },
]


def get_domain_code_map() -> dict:
    """Returns {code: domain_dict} for quick lookup."""
    return {d["code"]: d for d in CONTROL_DOMAINS}
