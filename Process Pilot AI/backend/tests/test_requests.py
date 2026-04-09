def _create_request(client, headers):
    payload = {
        "title": "Test Access Request",
        "description": "Need access to the reporting dashboard for quarterly analysis.",
        "category": "access_request",
        "urgency": 3,
        "business_impact": 3,
    }
    return client.post("/api/requests/", json=payload, headers=headers)


def test_create_request(client, employee_headers):
    resp = _create_request(client, employee_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Access Request"
    assert data["routing_decision"] is not None
    assert data["routing_decision"]["suggested_team"] == "IT_Support"
    assert data["priority_score"] is not None


def test_list_requests(client, employee_headers):
    _create_request(client, employee_headers)
    resp = client.get("/api/requests/", headers=employee_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "requester_name" in data[0]


def test_get_request_detail(client, employee_headers):
    create_resp = _create_request(client, employee_headers)
    request_id = create_resp.json()["id"]
    resp = client.get(f"/api/requests/{request_id}", headers=employee_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == request_id
    assert "routing_decision" in data
    assert "ai_summary" in data
    assert "updates" in data


def test_update_request_as_manager(client, manager_headers, employee_headers):
    create_resp = _create_request(client, employee_headers)
    request_id = create_resp.json()["id"]
    resp = client.patch(
        f"/api/requests/{request_id}",
        json={"status": "in_review", "note": "Reviewing this request now."},
        headers=manager_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "in_review"
    assert len(data["updates"]) >= 1


def test_update_request_as_employee_fails(client, employee_headers):
    create_resp = _create_request(client, employee_headers)
    request_id = create_resp.json()["id"]
    resp = client.patch(
        f"/api/requests/{request_id}",
        json={"status": "in_review"},
        headers=employee_headers,
    )
    assert resp.status_code == 403
