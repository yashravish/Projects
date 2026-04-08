from fastapi import APIRouter
from app.services.device_service import DeviceService
from app.schemas import DeviceStatusResponse

router = APIRouter(prefix="/api/device", tags=["device"])


@router.get("/status", response_model=DeviceStatusResponse)
def get_device_status():
    """Proxy the device simulator status endpoint."""
    status_data = DeviceService.get_status()
    return DeviceStatusResponse(
        device_name=status_data.get("device_name", "SIMULATED_SCANNER_01"),
        status=status_data.get("status", "unknown"),
        uptime_seconds=status_data.get("uptime_seconds"),
        firmware_version=status_data.get("firmware_version"),
        last_calibration=status_data.get("last_calibration"),
        capture_count=status_data.get("capture_count", 0),
        failure_mode=status_data.get("failure_mode"),
    )
