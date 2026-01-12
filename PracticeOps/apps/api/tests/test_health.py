"""Tests for health endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    """Health endpoint returns 200 with status ok."""
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "db" in data


@pytest.mark.asyncio
async def test_health_response_schema(client: AsyncClient) -> None:
    """Health endpoint returns expected schema with DB status."""
    response = await client.get("/health")

    data = response.json()
    assert isinstance(data["status"], str)
    assert isinstance(data["db"], str)
    assert data["db"] in ["ok", "error"]


@pytest.mark.asyncio
async def test_openapi_json_exposed(client: AsyncClient) -> None:
    """OpenAPI JSON is exposed at /openapi.json."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "PracticeOps API"

