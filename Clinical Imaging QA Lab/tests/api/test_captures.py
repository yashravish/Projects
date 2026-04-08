"""API tests for the captures endpoints."""
import pytest
import httpx

pytestmark = pytest.mark.api


class TestCapturesAPI:
    """Tests for /api/captures endpoints."""

    def test_create_capture_success(self, backend_url):
        """Verify a valid capture request returns 201 with capture data."""
        response = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "patient_id": "PAT-TEST-001",
                "session_id": "SESS-TEST-001",
                "image_type": "x-ray",
            },
            timeout=15.0,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["patient_id"] == "PAT-TEST-001"
        assert data["session_id"] == "SESS-TEST-001"
        assert data["image_type"] == "x-ray"
        assert data["capture_status"] in ("success", "failed")
        assert isinstance(data["id"], int)

    def test_create_capture_invalid_image_type(self, backend_url):
        """Verify invalid image type returns 422 validation error."""
        response = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "patient_id": "PAT-001",
                "session_id": "SESS-001",
                "image_type": "invalid-type",
            },
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_create_capture_missing_patient_id(self, backend_url):
        """Verify missing required field returns 422."""
        response = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "session_id": "SESS-001",
                "image_type": "mri",
            },
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_create_capture_empty_patient_id(self, backend_url):
        """Verify empty patient_id returns 422."""
        response = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "patient_id": "",
                "session_id": "SESS-001",
                "image_type": "mri",
            },
            timeout=10.0,
        )
        assert response.status_code == 422

    def test_list_captures(self, backend_url):
        """Verify captures list returns 200 with an array."""
        response = httpx.get(f"{backend_url}/api/captures", timeout=10.0)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_capture_by_id(self, backend_url):
        """Verify retrieving a single capture by ID."""
        create_resp = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "patient_id": "PAT-GET-001",
                "session_id": "SESS-GET-001",
                "image_type": "ct-scan",
            },
            timeout=15.0,
        )
        capture_id = create_resp.json()["id"]

        response = httpx.get(
            f"{backend_url}/api/captures/{capture_id}", timeout=10.0
        )
        assert response.status_code == 200
        assert response.json()["id"] == capture_id

    def test_get_capture_not_found(self, backend_url):
        """Verify 404 for nonexistent capture ID."""
        response = httpx.get(
            f"{backend_url}/api/captures/999999", timeout=10.0
        )
        assert response.status_code == 404

    def test_retry_capture(self, backend_url):
        """Verify retry endpoint processes a failed capture."""
        create_resp = httpx.post(
            f"{backend_url}/api/captures",
            json={
                "patient_id": "PAT-RETRY-001",
                "session_id": "SESS-RETRY-001",
                "image_type": "ultrasound",
            },
            timeout=15.0,
        )
        capture = create_resp.json()

        if capture["capture_status"] == "failed":
            retry_resp = httpx.post(
                f"{backend_url}/api/captures/{capture['id']}/retry",
                timeout=15.0,
            )
            assert retry_resp.status_code == 200
            retry_data = retry_resp.json()
            assert retry_data["retry_count"] >= 1

    def test_retry_nonexistent_capture(self, backend_url):
        """Verify 404 when retrying a nonexistent capture."""
        response = httpx.post(
            f"{backend_url}/api/captures/999999/retry", timeout=10.0
        )
        assert response.status_code == 404
