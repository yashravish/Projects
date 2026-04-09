def _create_request(client, headers, category="access_request"):
    payload = {
        "title": "Analytics Test Request",
        "description": "Request created for analytics testing purposes.",
        "category": category,
        "urgency": 3,
        "business_impact": 3,
    }
    return client.post("/api/requests/", json=payload, headers=headers)


def test_overview(client, employee_headers):
    _create_request(client, employee_headers)
    resp = client.get("/api/analytics/overview", headers=employee_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "open_requests" in data
    assert "closed_requests" in data
    assert "avg_priority" in data
    assert "requests_this_week" in data
    assert data["total_requests"] >= 1


def test_by_category(client, employee_headers):
    _create_request(client, employee_headers, category="workflow_issue")
    resp = client.get("/api/analytics/by-category", headers=employee_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "category" in data[0]
    assert "count" in data[0]
