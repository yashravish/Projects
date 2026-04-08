"""API tests for the dashboard summary endpoint."""
import pytest
import httpx

pytestmark = pytest.mark.api


class TestDashboardAPI:
    """Tests for /api/dashboard endpoints."""

    def test_dashboard_summary(self, backend_url):
        """Verify dashboard summary returns all expected fields."""
        response = httpx.get(
            f"{backend_url}/api/dashboard/summary", timeout=10.0
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_captures" in data
        assert "successful_captures" in data
        assert "failed_captures" in data
        assert "pending_captures" in data
        assert "total_defects" in data
        assert "open_defects" in data
        assert "device_status" in data
        assert "recent_captures" in data
        assert "recent_defects" in data
        assert isinstance(data["recent_captures"], list)
        assert isinstance(data["recent_defects"], list)

    def test_dashboard_counts_are_non_negative(self, backend_url):
        """Verify all count fields are non-negative integers."""
        response = httpx.get(
            f"{backend_url}/api/dashboard/summary", timeout=10.0
        )
        data = response.json()
        for field in [
            "total_captures", "successful_captures",
            "failed_captures", "pending_captures",
            "total_defects", "open_defects",
        ]:
            assert isinstance(data[field], int)
            assert data[field] >= 0

    def test_dashboard_recent_captures_limit(self, backend_url):
        """Verify recent captures list has at most 5 entries."""
        response = httpx.get(
            f"{backend_url}/api/dashboard/summary", timeout=10.0
        )
        data = response.json()
        assert len(data["recent_captures"]) <= 5

    def test_dashboard_recent_defects_limit(self, backend_url):
        """Verify recent defects list has at most 5 entries."""
        response = httpx.get(
            f"{backend_url}/api/dashboard/summary", timeout=10.0
        )
        data = response.json()
        assert len(data["recent_defects"]) <= 5

    def test_health_endpoint(self, backend_url):
        """Verify health endpoint returns healthy status."""
        response = httpx.get(f"{backend_url}/api/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
