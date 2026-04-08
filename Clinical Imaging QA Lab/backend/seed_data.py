"""Seed the database with sample data for development and demos."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal, init_db
from app.models import Capture, Defect, DeviceEvent
from datetime import datetime, timezone


def seed():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Capture).count() > 0:
            print("Database already has data — skipping seed.")
            return

        captures = [
            Capture(
                patient_id="PAT-001",
                session_id="SESS-1001",
                image_type="x-ray",
                capture_status="success",
                device_name="SIMULATED_SCANNER_01",
                device_response_code=200,
                file_path="/images/captures/PAT-001_xray_001.dcm",
                retry_count=0,
                captured_at=datetime.now(timezone.utc),
            ),
            Capture(
                patient_id="PAT-002",
                session_id="SESS-1002",
                image_type="mri",
                capture_status="failed",
                device_name="SIMULATED_SCANNER_01",
                device_response_code=500,
                retry_count=1,
                error_message="Device returned internal error during MRI capture",
            ),
            Capture(
                patient_id="PAT-003",
                session_id="SESS-1003",
                image_type="ct-scan",
                capture_status="success",
                device_name="SIMULATED_SCANNER_01",
                device_response_code=200,
                file_path="/images/captures/PAT-003_ct_001.dcm",
                retry_count=0,
                captured_at=datetime.now(timezone.utc),
            ),
        ]

        defects = [
            Defect(
                title="Capture timeout not displayed to user",
                severity="major",
                priority="high",
                environment="Chrome 120 / Windows 11",
                steps_to_reproduce="1. Set device to timeout mode\n2. Submit capture\n3. Observe UI",
                expected_result="Timeout error message shown in capture form",
                actual_result="Generic 'Something went wrong' shown instead of timeout detail",
                status="open",
            ),
            Defect(
                title="History table missing retry count column on mobile",
                severity="minor",
                priority="medium",
                environment="Safari iOS 17",
                steps_to_reproduce="1. Open history page on mobile viewport\n2. Check table columns",
                expected_result="Retry count column visible or accessible",
                actual_result="Retry count column hidden with no alternative access",
                status="open",
            ),
        ]

        events = [
            DeviceEvent(
                device_name="SIMULATED_SCANNER_01",
                event_type="status_check",
                details="Routine status poll — device online",
            ),
            DeviceEvent(
                device_name="SIMULATED_SCANNER_01",
                event_type="capture_success",
                details="Capture for PAT-001 completed successfully",
            ),
        ]

        db.add_all(captures + defects + events)
        db.commit()
        print(f"Seeded {len(captures)} captures, {len(defects)} defects, {len(events)} events.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
