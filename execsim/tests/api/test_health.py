"""Tests for health endpoints."""


def test_health_returns_ok(client):
    """GET /api/v1/health returns status ok."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_returns_ready(client):
    """GET /api/v1/ready returns status ready when DB is reachable."""
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
