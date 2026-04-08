import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Capture
from app.schemas import CaptureCreate
from app.services.device_service import DeviceService

logger = logging.getLogger(__name__)


class CaptureService:
    """Business logic for imaging capture operations."""

    @staticmethod
    def create_capture(db: Session, data: CaptureCreate) -> Capture:
        """Create a new capture by communicating with the device simulator."""
        capture = Capture(
            patient_id=data.patient_id,
            session_id=data.session_id,
            image_type=data.image_type,
            capture_status="pending",
            retry_count=0,
        )
        db.add(capture)
        db.commit()
        db.refresh(capture)

        DeviceService.log_event(
            db, "SIMULATED_SCANNER_01", "capture_attempt",
            f"Capture {capture.id} for patient {data.patient_id}"
        )

        result = DeviceService.request_capture(
            data.patient_id, data.session_id, data.image_type
        )
        status_code = result["status_code"]
        response_data = result["data"]

        if status_code == 200 and response_data.get("success"):
            capture.capture_status = "success"
            capture.device_name = response_data.get("device_name", "SIMULATED_SCANNER_01")
            capture.device_response_code = status_code
            capture.file_path = response_data.get("file_path")
            capture.captured_at = datetime.now(timezone.utc)
            capture.error_message = None
            DeviceService.log_event(
                db, capture.device_name, "capture_success",
                f"Capture {capture.id} succeeded"
            )
        else:
            capture.capture_status = "failed"
            capture.device_name = response_data.get("device_name", "SIMULATED_SCANNER_01")
            capture.device_response_code = status_code
            capture.error_message = response_data.get("error", "Unknown device error")
            DeviceService.log_event(
                db, capture.device_name or "SIMULATED_SCANNER_01", "capture_failure",
                f"Capture {capture.id} failed: {capture.error_message}"
            )

        db.commit()
        db.refresh(capture)
        logger.info(
            "Capture %d completed with status: %s", capture.id, capture.capture_status
        )
        return capture

    @staticmethod
    def list_captures(db: Session, limit: int = 100, offset: int = 0) -> list[Capture]:
        """Retrieve capture records ordered by most recent."""
        return (
            db.query(Capture)
            .order_by(Capture.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_capture(db: Session, capture_id: int) -> Capture | None:
        """Retrieve a single capture by ID."""
        return db.query(Capture).filter(Capture.id == capture_id).first()

    @staticmethod
    def retry_capture(db: Session, capture_id: int) -> Capture | None:
        """Retry a previously failed capture."""
        capture = db.query(Capture).filter(Capture.id == capture_id).first()
        if not capture:
            return None
        if capture.capture_status not in ("failed",):
            return capture

        capture.retry_count += 1
        capture.capture_status = "pending"
        capture.error_message = None
        db.commit()

        DeviceService.log_event(
            db, capture.device_name or "SIMULATED_SCANNER_01", "capture_retry",
            f"Capture {capture.id} retry #{capture.retry_count}"
        )

        result = DeviceService.request_capture(
            capture.patient_id, capture.session_id, capture.image_type
        )
        status_code = result["status_code"]
        response_data = result["data"]

        if status_code == 200 and response_data.get("success"):
            capture.capture_status = "success"
            capture.device_response_code = status_code
            capture.file_path = response_data.get("file_path")
            capture.captured_at = datetime.now(timezone.utc)
            capture.error_message = None
        else:
            capture.capture_status = "failed"
            capture.device_response_code = status_code
            capture.error_message = response_data.get("error", "Unknown device error")

        db.commit()
        db.refresh(capture)
        logger.info(
            "Capture %d retry completed with status: %s",
            capture.id, capture.capture_status
        )
        return capture

    @staticmethod
    def get_summary(db: Session) -> dict:
        """Return aggregate counts for the dashboard."""
        total = db.query(func.count(Capture.id)).scalar() or 0
        success = (
            db.query(func.count(Capture.id))
            .filter(Capture.capture_status == "success")
            .scalar() or 0
        )
        failed = (
            db.query(func.count(Capture.id))
            .filter(Capture.capture_status == "failed")
            .scalar() or 0
        )
        pending = (
            db.query(func.count(Capture.id))
            .filter(Capture.capture_status == "pending")
            .scalar() or 0
        )
        return {
            "total_captures": total,
            "successful_captures": success,
            "failed_captures": failed,
            "pending_captures": pending,
        }
