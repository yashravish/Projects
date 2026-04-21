"""Tests for opportunity and fill API endpoints."""


def test_list_opportunities(client):
    """GET /api/v1/runs/{id}/opportunities returns opportunities."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 100})
    run_id = create_resp.json()["id"]

    resp = client.get(f"/api/v1/runs/{run_id}/opportunities")
    assert resp.status_code == 200
    opps = resp.json()
    assert isinstance(opps, list)


def test_list_fills(client):
    """GET /api/v1/runs/{id}/fills returns fills."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 100})
    run_id = create_resp.json()["id"]

    resp = client.get(f"/api/v1/runs/{run_id}/fills")
    assert resp.status_code == 200
    fills = resp.json()
    assert isinstance(fills, list)


def test_get_run_metrics(client):
    """GET /api/v1/runs/{id}/metrics returns aggregated metrics."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 100})
    run_id = create_resp.json()["id"]

    resp = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert "num_opportunities" in data
    assert "num_fills" in data
