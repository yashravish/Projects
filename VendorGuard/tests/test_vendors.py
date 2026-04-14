"""Tests for vendor CRUD operations."""


def test_create_vendor(client, auth_headers):
    vendor_data = {
        "name": "Test SaaS Platform",
        "category": "SaaS",
        "description": "A test vendor",
        "hosting_model": "cloud",
        "deployment_scope": "enterprise",
        "internet_exposed": True,
        "handles_sensitive_data": True,
        "data_types": ["PII"],
        "compliance_attestations": ["SOC 2 Type II"],
    }
    res = client.post("/api/vendors", json=vendor_data, headers=auth_headers)
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == "Test SaaS Platform"
    assert data["category"] == "SaaS"
    assert data["handles_sensitive_data"] is True


def test_create_vendor_invalid_category(client, auth_headers):
    res = client.post("/api/vendors", json={"name": "Bad", "category": "Invalid"}, headers=auth_headers)
    assert res.status_code == 400


def test_list_vendors(client, auth_headers):
    res = client.get("/api/vendors", headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_get_vendor(client, auth_headers):
    create = client.post("/api/vendors", json={
        "name": "Detail Vendor", "category": "PaaS",
    }, headers=auth_headers)
    vendor_id = create.json()["id"]

    res = client.get(f"/api/vendors/{vendor_id}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["name"] == "Detail Vendor"


def test_get_vendor_not_found(client, auth_headers):
    res = client.get("/api/vendors/99999", headers=auth_headers)
    assert res.status_code == 404


def test_create_integration(client, auth_headers):
    create = client.post("/api/vendors", json={
        "name": "Integ Vendor", "category": "SaaS",
    }, headers=auth_headers)
    vendor_id = create.json()["id"]

    res = client.post(f"/api/vendors/{vendor_id}/integrations", json={
        "system_name": "Active Directory",
        "integration_type": "SSO",
        "data_flow_direction": "inbound",
    }, headers=auth_headers)
    assert res.status_code == 201
    assert res.json()["system_name"] == "Active Directory"


def test_vendor_requires_auth(client):
    res = client.get("/api/vendors")
    assert res.status_code == 401
