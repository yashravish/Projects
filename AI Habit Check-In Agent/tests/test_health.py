"""Tests for the health check endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_check_returns_200(client):
    """GET /health should return 200 with healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-habit-checkin-agent"


@pytest.mark.asyncio
async def test_health_check_response_format(client):
    """GET /health response should contain exactly the expected keys."""
    response = await client.get("/health")
    data = response.json()

    assert set(data.keys()) == {"status", "service"}
