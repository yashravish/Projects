import logging
import httpx
from app.config import settings
from app.models import DeviceEvent
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DeviceService:
    """Handles communication with the mock device simulator."""

    @staticmethod
    def _base_url() -> str:
        return settings.device_simulator_url

    @staticmethod
    def get_status() -> dict:
        """Fetch current device status from the simulator."""
        try:
            response = httpx.get(
                f"{DeviceService._base_url()}/device/status", timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            logger.error("Device simulator is unreachable")
            return {"device_name": "SIMULATED_SCANNER_01", "status": "unreachable"}
        except httpx.HTTPStatusError as exc:
            logger.error("Device status returned %s", exc.response.status_code)
            return {"device_name": "SIMULATED_SCANNER_01", "status": "error"}
        except Exception as exc:
            logger.exception("Unexpected error fetching device status")
            return {"device_name": "SIMULATED_SCANNER_01", "status": "unknown"}

    @staticmethod
    def request_capture(patient_id: str, session_id: str, image_type: str) -> dict:
        """Request a capture from the device simulator."""
        try:
            response = httpx.post(
                f"{DeviceService._base_url()}/device/capture",
                json={
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "image_type": image_type,
                },
                timeout=10.0,
            )
            return {
                "status_code": response.status_code,
                "data": response.json(),
            }
        except httpx.ConnectError:
            logger.error("Device simulator unreachable during capture")
            return {
                "status_code": 503,
                "data": {"error": "Device simulator unreachable"},
            }
        except httpx.ReadTimeout:
            logger.error("Device capture request timed out")
            return {
                "status_code": 504,
                "data": {"error": "Device capture timed out"},
            }
        except Exception as exc:
            logger.exception("Unexpected error during capture request")
            return {
                "status_code": 500,
                "data": {"error": str(exc)},
            }

    @staticmethod
    def log_event(db: Session, device_name: str, event_type: str, details: str = None):
        """Record a device interaction in the audit log."""
        event = DeviceEvent(
            device_name=device_name, event_type=event_type, details=details
        )
        db.add(event)
        db.commit()
        logger.info("Device event logged: %s - %s", event_type, details or "")
