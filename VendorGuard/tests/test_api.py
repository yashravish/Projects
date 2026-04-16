"""Integration tests for assessment workflow and API endpoints."""


def _create_test_vendor(client, auth_headers, name="API Test Vendor"):
    return client.post("/api/vendors", json={
        "name": name,
        "category": "SaaS",
        "hosting_model": "cloud",
        "internet_exposed": True,
        "handles_sensitive_data": True,
    }, headers=auth_headers)


def test_full_assessment_workflow(client, auth_headers):
    """End-to-end: create vendor -> create assessment -> submit answers -> evaluate."""
    vendor_res = _create_test_vendor(client, auth_headers, "Workflow Test Vendor")
    assert vendor_res.status_code == 201
    vendor_id = vendor_res.json()["id"]

    assessment_res = client.post("/api/assessments", json={
        "vendor_id": vendor_id,
        "assessment_type": "initial",
        "phase": "pre_implementation",
    }, headers=auth_headers)
    assert assessment_res.status_code == 201
    assessment_id = assessment_res.json()["id"]
    assert assessment_res.json()["status"] == "draft"

    answers = [
        {"question_key": "encryption_rest", "section": "encryption", "answer": "false"},
        {"question_key": "encryption_transit", "section": "encryption", "answer": "true"},
        {"question_key": "data_classification", "section": "data_handling", "answer": "Confidential"},
        {"question_key": "mfa_supported", "section": "identity_access", "answer": "false"},
        {"question_key": "privileged_access_required", "section": "identity_access", "answer": "true"},
        {"question_key": "audit_logging", "section": "logging_monitoring", "answer": "true"},
        {"question_key": "log_export", "section": "logging_monitoring", "answer": "false"},
        {"question_key": "ir_plan_documented", "section": "incident_response", "answer": "true"},
        {"question_key": "breach_notification_commitment", "section": "incident_response", "answer": "false"},
        {"question_key": "vuln_mgmt_program", "section": "vulnerability_mgmt", "answer": "true"},
        {"question_key": "patching_cadence", "section": "vulnerability_mgmt", "answer": "Within 7 days"},
        {"question_key": "soc2_certified", "section": "compliance", "answer": "false"},
        {"question_key": "iso27001_certified", "section": "compliance", "answer": "false"},
        {"question_key": "backup_procedures", "section": "business_continuity", "answer": "true"},
        {"question_key": "dr_plan", "section": "business_continuity", "answer": "true"},
        {"question_key": "subprocessors_documented", "section": "compliance", "answer": "true"},
        {"question_key": "sso_supported", "section": "identity_access", "answer": "true"},
        {"question_key": "integration_count", "section": "integrations", "answer": "1-3"},
    ]

    submit_res = client.post(f"/api/assessments/{assessment_id}/submit",
                             json={"answers": answers}, headers=auth_headers)
    assert submit_res.status_code == 200

    eval_res = client.post(f"/api/assessments/{assessment_id}/evaluate", headers=auth_headers)
    assert eval_res.status_code == 200
    result = eval_res.json()
    assert "inherent_risk_score" in result
    assert "findings_count" in result
    assert result["findings_count"] > 0

    get_res = client.get(f"/api/assessments/{assessment_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["status"] == "completed"


def test_list_assessments(client, auth_headers):
    res = client.get("/api/assessments", headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_findings_endpoint(client, auth_headers):
    res = client.get("/api/findings", headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_findings_filter_by_severity(client, auth_headers):
    res = client.get("/api/findings?severity=High", headers=auth_headers)
    assert res.status_code == 200
    for f in res.json():
        assert f["severity"] == "High"


def test_remediation_list(client, auth_headers):
    res = client.get("/api/remediation", headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_remediation_update(client, auth_headers):
    items = client.get("/api/remediation", headers=auth_headers).json()
    if items:
        item_id = items[0]["id"]
        res = client.patch(f"/api/remediation/{item_id}",
                           json={"status": "in_progress", "assigned_to": "Test User"},
                           headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "in_progress"


def test_dashboard(client, auth_headers):
    res = client.get("/api/dashboard", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "total_vendors" in data
    assert "findings_by_severity" in data


def test_templates_list(client, auth_headers):
    res = client.get("/api/templates", headers=auth_headers)
    assert res.status_code == 200


def test_control_domains_list(client, auth_headers):
    res = client.get("/api/templates/domains/list", headers=auth_headers)
    assert res.status_code == 200
    domains = res.json()
    assert len(domains) >= 12


def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_report_requires_completed_assessment(client, auth_headers):
    vendor_res = _create_test_vendor(client, auth_headers, "Report Test Vendor")
    vendor_id = vendor_res.json()["id"]
    assessment_res = client.post("/api/assessments", json={
        "vendor_id": vendor_id, "assessment_type": "initial", "phase": "pre_implementation",
    }, headers=auth_headers)
    assessment_id = assessment_res.json()["id"]
    res = client.get(f"/api/reports/{assessment_id}", headers=auth_headers)
    assert res.status_code == 400
