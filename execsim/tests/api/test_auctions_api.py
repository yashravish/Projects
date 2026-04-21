"""Tests for auction API endpoints."""

import uuid


def test_create_auction(client):
    """POST /api/v1/runs/{id}/auction runs auction and returns result."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 100})
    run_id = create_resp.json()["id"]

    resp = client.post(
        f"/api/v1/runs/{run_id}/auction",
        json={"reserve_price_bps": 5.0, "n_bidders": 3},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == run_id
    assert "num_opportunities" in data
    assert "num_allocated" in data


def test_auction_on_missing_run(client):
    """POST auction on non-existent run returns 404."""
    fake_id = str(uuid.uuid4())
    resp = client.post(
        f"/api/v1/runs/{fake_id}/auction",
        json={"reserve_price_bps": 0},
    )
    assert resp.status_code == 404


def test_get_auction_detail(client):
    """GET /api/v1/auctions/{id} returns auction detail with entries."""
    create_resp = client.post("/api/v1/runs", json={"seed": 42, "num_steps": 50})
    run_id = create_resp.json()["id"]

    auction_resp = client.post(
        f"/api/v1/runs/{run_id}/auction",
        json={"reserve_price_bps": 0, "n_bidders": 2},
    )
    auction_id = auction_resp.json()["id"]

    resp = client.get(f"/api/v1/auctions/{auction_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == auction_id
    assert "entries" in data
    assert "result" in data


def test_calibrate(client):
    """POST /api/v1/calibrate returns calibration result."""
    resp = client.post("/api/v1/calibrate", json={
        "training_seeds": [1, 2],
        "held_out_seeds": [3],
        "n_bidders": 2,
        "grid_max_bps": 10,
        "grid_step_bps": 5,
        "allocation_floor": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "optimal_reserve_bps" in data
    assert "grid" in data
    assert len(data["grid"]) > 0
