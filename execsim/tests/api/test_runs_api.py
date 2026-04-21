"""Tests for run API endpoints."""

import uuid


def test_create_run(client):
    """POST /api/v1/runs creates a run and returns it."""
    resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 50})
    assert resp.status_code == 201
    data = resp.json()
    assert data["seed"] == 42
    assert data["status"] == "completed"
    assert "id" in data


def test_list_runs(client):
    """GET /api/v1/runs returns a list."""
    client.post("/api/v1/runs", json={"seed": 1, "num_steps": 20})
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1


def test_get_run_detail(client):
    """GET /api/v1/runs/{id} returns full detail."""
    create_resp = client.post("/api/v1/runs", json={"seed": 2, "num_steps": 20})
    run_id = create_resp.json()["id"]

    resp = client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == run_id
    assert data["num_steps"] == 20
    assert "num_opportunities" in data


def test_get_run_not_found(client):
    """GET /api/v1/runs/{id} returns 404 for unknown id."""
    fake_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/runs/{fake_id}")
    assert resp.status_code == 404
