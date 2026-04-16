"""Tests for the deterministic risk engine, rules, and scoring."""

from backend.engine.rules import (
    evaluate_rules,
    rule_encryption_at_rest,
    rule_no_mfa_privileged,
    rule_no_audit_logging,
    rule_ai_broad_data_access,
    rule_iot_no_segmentation,
    rule_iot_default_credentials,
    rule_token_no_key_management,
    rule_dlt_pii_no_privacy,
    rule_sw_local_admin_no_update,
)
from backend.engine.scoring import (
    calculate_inherent_risk,
    calculate_residual_risk,
    severity_to_value,
    score_to_rating,
)


# --- Scoring Tests ---

def test_severity_values():
    assert severity_to_value("Critical") == 4
    assert severity_to_value("High") == 3
    assert severity_to_value("Moderate") == 2
    assert severity_to_value("Low") == 1
    assert severity_to_value("Unknown") == 1


def test_score_to_rating():
    assert score_to_rating(0) == "Low"
    assert score_to_rating(25) == "Low"
    assert score_to_rating(26) == "Moderate"
    assert score_to_rating(50) == "Moderate"
    assert score_to_rating(51) == "High"
    assert score_to_rating(75) == "High"
    assert score_to_rating(76) == "Critical"
    assert score_to_rating(100) == "Critical"


def test_calculate_inherent_risk_no_findings():
    score, rating = calculate_inherent_risk([])
    assert score == 0.0
    assert rating == "Low"


def test_calculate_inherent_risk_with_findings():
    findings = [
        {"severity": "Critical", "domain_code": "DP"},
        {"severity": "High", "domain_code": "AC"},
        {"severity": "Moderate", "domain_code": "LM"},
    ]
    score, rating = calculate_inherent_risk(findings)
    # Critical(4)*DP(5) + High(3)*AC(5) + Moderate(2)*LM(4) = 20+15+8 = 43
    # 43/200*100 = 21.5
    assert score == 21.5
    assert rating == "Low"


def test_calculate_residual_risk():
    findings = [
        {"severity": "High", "domain_code": "AC", "remediation_status": "mitigated"},
        {"severity": "High", "domain_code": "DP", "remediation_status": "open"},
    ]
    inherent = 30.0
    residual_score, _ = calculate_residual_risk(inherent, findings)
    assert residual_score < inherent


# --- Rule Tests ---

def test_rule_encryption_at_rest_triggers():
    vendor = {"handles_sensitive_data": True}
    answers = {"encryption_rest": "false", "data_classification": "Restricted/Regulated"}
    result = rule_encryption_at_rest(vendor, answers)
    assert result is not None
    assert result["severity"] == "Critical"
    assert result["domain_code"] == "DP"


def test_rule_encryption_at_rest_no_trigger():
    vendor = {"handles_sensitive_data": True}
    answers = {"encryption_rest": "true"}
    result = rule_encryption_at_rest(vendor, answers)
    assert result is None


def test_rule_no_mfa_privileged_triggers():
    vendor = {}
    answers = {"privileged_access_required": "true", "mfa_supported": "false"}
    result = rule_no_mfa_privileged(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_rule_no_audit_logging_sensitive():
    vendor = {"handles_sensitive_data": True}
    answers = {"audit_logging": "false"}
    result = rule_no_audit_logging(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_rule_ai_broad_data_access():
    vendor = {"category": "AI Tool"}
    answers = {
        "ai_data_access_scope": "Broad organizational data",
        "ai_prompt_retention": "Retained for model training",
    }
    result = rule_ai_broad_data_access(vendor, answers)
    assert result is not None
    assert result["severity"] == "Critical"
    assert result["domain_code"] == "AG"


def test_rule_ai_not_triggered_for_saas():
    vendor = {"category": "SaaS"}
    answers = {"ai_data_access_scope": "Broad organizational data"}
    result = rule_ai_broad_data_access(vendor, answers)
    assert result is None


def test_rule_iot_no_segmentation():
    vendor = {"category": "IoT Platform"}
    answers = {"iot_network_segmentation": "false"}
    result = rule_iot_no_segmentation(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_rule_iot_default_credentials():
    vendor = {"category": "IoT Platform"}
    answers = {"iot_default_credentials": "false"}
    result = rule_iot_default_credentials(vendor, answers)
    assert result is not None
    assert result["severity"] == "Critical"


def test_rule_token_no_key_management():
    vendor = {"category": "Tokenization Platform"}
    answers = {"token_key_management": "false"}
    result = rule_token_no_key_management(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_rule_dlt_pii_no_privacy():
    vendor = {"category": "Distributed Ledger Platform"}
    answers = {"dlt_pii_on_chain": "true", "dlt_privacy_controls": "false"}
    result = rule_dlt_pii_no_privacy(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_rule_sw_local_admin_no_update():
    vendor = {"category": "End-User Software Package"}
    answers = {"sw_local_admin_required": "true", "sw_auto_update": "false"}
    result = rule_sw_local_admin_no_update(vendor, answers)
    assert result is not None
    assert result["severity"] == "High"


def test_evaluate_rules_comprehensive():
    """Test that the full engine produces expected findings for a risky AI tool."""
    vendor = {
        "category": "AI Tool",
        "handles_sensitive_data": True,
        "internet_exposed": True,
        "deployment_scope": "enterprise",
    }
    answers = {
        "encryption_rest": "false",
        "encryption_transit": "true",
        "data_classification": "Confidential",
        "mfa_supported": "false",
        "privileged_access_required": "true",
        "sso_supported": "false",
        "audit_logging": "false",
        "ir_plan_documented": "false",
        "vuln_mgmt_program": "false",
        "soc2_certified": "false",
        "iso27001_certified": "false",
        "backup_procedures": "false",
        "dr_plan": "false",
        "subprocessors_documented": "false",
        "ai_data_access_scope": "Broad organizational data",
        "ai_prompt_retention": "Retained for model training",
        "ai_model_training_on_data": "true",
        "ai_admin_controls": "false",
        "integration_count": "4-10",
        "patching_cadence": "Unknown",
    }
    findings = evaluate_rules(vendor, answers)
    assert len(findings) >= 10
    severities = [f["severity"] for f in findings]
    assert "Critical" in severities
    assert "High" in severities
