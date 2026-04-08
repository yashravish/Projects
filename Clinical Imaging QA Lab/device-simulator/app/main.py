"""Mock imaging device simulator — standalone FastAPI service."""
import random
import time
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from app.device_state import device_state
from app.schemas import CaptureRequest, FailureModeRequest

app = FastAPI(
    title="Device Simulator",
    description="Simulated clinical imaging hardware device",
    version="1.0.0",
)


@app.get("/device/status")
def get_status():
    """Return current device status and metadata."""
    return device_state.to_dict()


@app.post("/device/capture")
def capture_image(request: CaptureRequest):
    """Simulate an image capture with configurable failure behavior."""
    if device_state.status == "offline":
        return {
            "success": False,
            "device_name": "SIMULATED_SCANNER_01",
            "error": "Device is offline",
            "error_code": "DEVICE_OFFLINE",
        }

    failure = device_state.failure_mode

    if failure == "timeout":
        time.sleep(6)
        return {
            "success": False,
            "device_name": "SIMULATED_SCANNER_01",
            "error": "Capture timed out",
            "error_code": "TIMEOUT",
        }

    if failure == "unavailable":
        raise HTTPException(status_code=503, detail="Device temporarily unavailable")

    if failure == "random_failure":
        if random.random() < 0.5:
            device_state.increment_capture()
            return _success_response(request)
        return {
            "success": False,
            "device_name": "SIMULATED_SCANNER_01",
            "error": "Random device malfunction",
            "error_code": "RANDOM_FAILURE",
        }

    if failure == "corrupted_metadata":
        device_state.increment_capture()
        return {
            "success": True,
            "device_name": "SIMULATED_SCANNER_01",
            "file_path": None,
            "capture_id": None,
            "timestamp": "INVALID_TIMESTAMP",
            "metadata": {"corrupted": True},
        }

    device_state.increment_capture()
    return _success_response(request)


def _success_response(request: CaptureRequest) -> dict:
    """Build a successful capture response with realistic metadata."""
    capture_uuid = uuid.uuid4().hex[:12]
    file_name = (
        f"{request.patient_id}_{request.image_type}_{capture_uuid}.dcm"
    )
    return {
        "success": True,
        "device_name": "SIMULATED_SCANNER_01",
        "capture_id": capture_uuid,
        "file_path": f"/images/captures/{file_name}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "patient_id": request.patient_id,
            "session_id": request.session_id,
            "image_type": request.image_type,
            "resolution": "2048x2048",
            "bit_depth": 16,
            "modality": request.image_type.upper().replace("-", "_"),
        },
    }


@app.post("/device/disconnect")
def disconnect():
    """Simulate device going offline."""
    device_state.status = "offline"
    return {"message": "Device disconnected", "status": "offline"}


@app.post("/device/reconnect")
def reconnect():
    """Simulate device coming back online."""
    device_state.status = "online"
    device_state.failure_mode = None
    return {"message": "Device reconnected", "status": "online"}


@app.post("/device/failure-mode")
def set_failure_mode(request: FailureModeRequest):
    """Configure a failure mode for the simulator.

    Accepted modes: timeout, random_failure, corrupted_metadata, unavailable, or null/empty to clear.
    """
    allowed = {None, "", "timeout", "random_failure", "corrupted_metadata", "unavailable"}
    mode = request.mode if request.mode else None
    if mode not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid failure mode. Allowed: {', '.join(str(m) for m in allowed if m)}"
        )
    device_state.failure_mode = mode
    return {
        "message": f"Failure mode set to: {mode or 'none'}",
        "failure_mode": mode,
    }


@app.post("/device/reset")
def reset_device():
    """Reset device to default state — useful for tests."""
    device_state.reset()
    return {"message": "Device reset to defaults", "status": "online"}


@app.get("/health")
def health():
    """Health check for the simulator service."""
    return {"status": "healthy", "service": "device-simulator"}
