"""API tests for the defects endpoints."""
import pytest
import httpx

pytestmark = pytest.mark.api


class TestDefectsAPI:
    """Tests for /api/defects endpoints."""

    def test_create_defect(self, backend_url):
        """Verify creating a defect returns 201 with correct data."""
        response = httpx.post(
            f"{backend_url}/api/defects",
            json={
                "title": "Test defect from API tests",
                "severity": "minor",
                "priority": "low",
                "environment": "Test Environment",
                "steps_to_reproduce": "1. Run test\n2. Observe",
                "expected_result": "No defect",
                "actual_result": "Defect found",
            },
            timeout=10.0,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test defect from API tests"
        assert data["severity"] == "minor"
        assert data["priority"] == "low"
        assert data["status"] == "open"
        assert isinstance(data["id"], int)

    def test_create_defect_missing_title(self, backend_url):
        """Verify missing title returns 422."""
        response = httpx.post(
            f"{backend_url}/api/defects",
            json={"severity": "major", "priority": "high"},
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_create_defect_invalid_severity(self, backend_url):
        """Verify invalid severity returns 422."""
        response = httpx.post(
            f"{backend_url}/api/defects",
            json={
                "title": "Invalid severity test",
                "severity": "ultra-critical",
                "priority": "high",
            },
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_create_defect_invalid_priority(self, backend_url):
        """Verify invalid priority returns 422."""
        response = httpx.post(
            f"{backend_url}/api/defects",
            json={
                "title": "Invalid priority test",
                "severity": "major",
                "priority": "urgent",
            },
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_list_defects(self, backend_url):
        """Verify defects list returns 200 with an array."""
        response = httpx.get(f"{backend_url}/api/defects", timeout=10.0)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_defect_by_id(self, backend_url):
        """Verify retrieving a single defect by ID."""
        create_resp = httpx.post(
            f"{backend_url}/api/defects",
            json={
                "title": "Defect for GET test",
                "severity": "trivial",
                "priority": "low",
            },
            timeout=10.0,
        )
        defect_id = create_resp.json()["id"]

        response = httpx.get(
            f"{backend_url}/api/defects/{defect_id}", timeout=10.0
        )
        assert response.status_code == 200
        assert response.json()["id"] == defect_id

    def test_get_defect_not_found(self, backend_url):
        """Verify 404 for nonexistent defect ID."""
        response = httpx.get(
            f"{backend_url}/api/defects/999999", timeout=10.0
        )
        assert response.status_code == 404

    def test_create_defect_minimal(self, backend_url):
        """Verify creating a defect with only required fields."""
        response = httpx.post(
            f"{backend_url}/api/defects",
            json={
                "title": "Minimal defect",
                "severity": "critical",
                "priority": "high",
            },
            timeout=10.0,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["environment"] is None
        assert data["steps_to_reproduce"] is None
