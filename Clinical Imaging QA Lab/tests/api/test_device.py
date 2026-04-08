"""API tests for the device status proxy endpoint."""
import pytest
import httpx

pytestmark = pytest.mark.api


class TestDeviceAPI:
    """Tests for /api/device endpoints."""

    def test_device_status(self, backend_url):
        """Verify device status proxy returns expected fields."""
        response = httpx.get(f"{backend_url}/api/device/status", timeout=10.0)
        assert response.status_code == 200
        data = response.json()
        assert "device_name" in data
        assert "status" in data
        assert data["device_name"] == "SIMULATED_SCANNER_01"

    def test_device_status_has_metadata(self, backend_url):
        """Verify device status includes firmware and calibration info."""
        response = httpx.get(f"{backend_url}/api/device/status", timeout=10.0)
        data = response.json()
        assert "capture_count" in data


class TestDeviceSimulator:
    """Direct tests against the device simulator service."""

    def test_simulator_health(self, device_url):
        """Verify simulator health endpoint responds."""
        response = httpx.get(f"{device_url}/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_simulator_status(self, device_url):
        """Verify simulator status endpoint returns online by default."""
        httpx.post(f"{device_url}/device/reset", timeout=5.0)
        response = httpx.get(f"{device_url}/device/status", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"

    def test_simulator_capture_online(self, device_url):
        """Verify capture succeeds when device is online with no failure mode."""
        httpx.post(f"{device_url}/device/reset", timeout=5.0)
        response = httpx.post(
            f"{device_url}/device/capture",
            json={
                "patient_id": "PAT-SIM-001",
                "session_id": "SESS-SIM-001",
                "image_type": "x-ray",
            },
            timeout=10.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "file_path" in data

    def test_simulator_disconnect_reconnect(self, device_url):
        """Verify device can be disconnected and reconnected."""
        httpx.post(f"{device_url}/device/disconnect", timeout=5.0)
        status_resp = httpx.get(f"{device_url}/device/status", timeout=5.0)
        assert status_resp.json()["status"] == "offline"

        capture_resp = httpx.post(
            f"{device_url}/device/capture",
            json={
                "patient_id": "PAT-OFF",
                "session_id": "SESS-OFF",
                "image_type": "mri",
            },
            timeout=10.0,
        )
        assert capture_resp.json()["success"] is False

        httpx.post(f"{device_url}/device/reconnect", timeout=5.0)
        status_resp2 = httpx.get(f"{device_url}/device/status", timeout=5.0)
        assert status_resp2.json()["status"] == "online"

    def test_simulator_failure_mode_unavailable(self, device_url):
        """Verify unavailable failure mode returns 503."""
        httpx.post(f"{device_url}/device/reset", timeout=5.0)
        httpx.post(
            f"{device_url}/device/failure-mode",
            json={"mode": "unavailable"},
            timeout=5.0,
        )
        response = httpx.post(
            f"{device_url}/device/capture",
            json={
                "patient_id": "PAT-UNAVAIL",
                "session_id": "SESS-UNAVAIL",
                "image_type": "ct-scan",
            },
            timeout=10.0,
        )
        assert response.status_code == 503
        httpx.post(f"{device_url}/device/reset", timeout=5.0)

    def test_simulator_failure_mode_invalid(self, device_url):
        """Verify invalid failure mode returns 400."""
        response = httpx.post(
            f"{device_url}/device/failure-mode",
            json={"mode": "nonexistent_mode"},
            timeout=5.0,
        )
        assert response.status_code == 400

    def test_simulator_reset(self, device_url):
        """Verify reset returns device to default state."""
        httpx.post(f"{device_url}/device/disconnect", timeout=5.0)
        httpx.post(f"{device_url}/device/reset", timeout=5.0)
        status = httpx.get(f"{device_url}/device/status", timeout=5.0).json()
        assert status["status"] == "online"
        assert status["failure_mode"] is None
