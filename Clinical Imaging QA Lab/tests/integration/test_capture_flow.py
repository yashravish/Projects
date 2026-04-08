"""Integration tests for end-to-end capture flow across all services."""
import pytest
import httpx

pytestmark = pytest.mark.integration

BACKEND_URL = "http://localhost:8000"
DEVICE_URL = "http://localhost:8001"


class TestCaptureFlow:
    """End-to-end integration tests for the capture workflow."""

    def test_full_capture_success_flow(self):
        """Verify the complete capture flow: reset device → capture → verify in history."""
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)

        resp = httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-INTEG-001",
                "session_id": "SESS-INTEG-001",
                "image_type": "x-ray",
            },
            timeout=15.0,
        )
        assert resp.status_code == 201
        capture = resp.json()
        assert capture["capture_status"] == "success"
        assert capture["file_path"] is not None

        history = httpx.get(f"{BACKEND_URL}/api/captures", timeout=10.0).json()
        found = [c for c in history if c["id"] == capture["id"]]
        assert len(found) == 1
        assert found[0]["capture_status"] == "success"

    def test_capture_with_device_offline(self):
        """Verify capture fails when device is offline."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        resp = httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-INTEG-002",
                "session_id": "SESS-INTEG-002",
                "image_type": "mri",
            },
            timeout=15.0,
        )
        assert resp.status_code == 201
        capture = resp.json()
        assert capture["capture_status"] == "failed"
        assert capture["error_message"] is not None

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)

    def test_capture_retry_after_device_reconnect(self):
        """Verify retry succeeds after device is reconnected."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        resp = httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-INTEG-003",
                "session_id": "SESS-INTEG-003",
                "image_type": "ct-scan",
            },
            timeout=15.0,
        )
        capture = resp.json()
        assert capture["capture_status"] == "failed"

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)

        retry_resp = httpx.post(
            f"{BACKEND_URL}/api/captures/{capture['id']}/retry",
            timeout=15.0,
        )
        retry_data = retry_resp.json()
        assert retry_data["retry_count"] >= 1
        assert retry_data["capture_status"] == "success"

    def test_defect_logging_flow(self):
        """Verify defect creation and retrieval in the defect tracker."""
        resp = httpx.post(
            f"{BACKEND_URL}/api/defects",
            json={
                "title": "Integration test defect",
                "severity": "major",
                "priority": "high",
                "environment": "Integration test suite",
                "steps_to_reproduce": "1. Run integration tests",
                "expected_result": "All pass",
                "actual_result": "Defect logged",
            },
            timeout=10.0,
        )
        assert resp.status_code == 201
        defect = resp.json()

        defects = httpx.get(f"{BACKEND_URL}/api/defects", timeout=10.0).json()
        found = [d for d in defects if d["id"] == defect["id"]]
        assert len(found) == 1
        assert found[0]["status"] == "open"

    def test_dashboard_reflects_data(self):
        """Verify dashboard summary reflects actual captures and defects."""
        summary = httpx.get(
            f"{BACKEND_URL}/api/dashboard/summary", timeout=10.0
        ).json()
        assert summary["total_captures"] >= 0
        assert summary["total_defects"] >= 0
        assert summary["total_captures"] == (
            summary["successful_captures"]
            + summary["failed_captures"]
            + summary["pending_captures"]
        )
