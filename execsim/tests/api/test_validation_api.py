"""Tests for validation API endpoints."""


def test_validate_run(client):
    """POST /api/v1/runs/{id}/validate returns validation alerts."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 50})
    run_id = create_resp.json()["id"]

    resp = client.post(f"/api/v1/runs/{run_id}/validate")
    assert resp.status_code == 200
    alerts = resp.json()
    assert isinstance(alerts, list)


def test_list_alerts(client):
    """GET /api/v1/alerts returns alerts."""
    # Create a run and validate it first
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 20})
    run_id = create_resp.json()["id"]
    client.post(f"/api/v1/runs/{run_id}/validate")

    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
