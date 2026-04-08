"""SQL validation tests — verify database state after key workflows."""
import pytest
import httpx
from tests.conftest import run_sql, count_rows

pytestmark = pytest.mark.integration

BACKEND_URL = "http://localhost:8000"
DEVICE_URL = "http://localhost:8001"


class TestSQLValidation:
    """Direct database assertions verifying data integrity after actions."""

    def test_successful_capture_inserts_row(self, db_session):
        """Assert a successful capture creates a row in the captures table."""
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)

        before = count_rows(db_session, "captures")
        httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-SQL-001",
                "session_id": "SESS-SQL-001",
                "image_type": "x-ray",
            },
            timeout=15.0,
        )
        db_session.commit()
        after = count_rows(db_session, "captures")
        assert after == before + 1

    def test_failed_capture_stores_error_message(self, db_session):
        """Assert a failed capture stores an error_message in the row."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        resp = httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-SQL-002",
                "session_id": "SESS-SQL-002",
                "image_type": "mri",
            },
            timeout=15.0,
        )
        capture_id = resp.json()["id"]

        rows = run_sql(
            db_session,
            "SELECT error_message, capture_status FROM captures WHERE id = :id",
            {"id": capture_id},
        )
        assert len(rows) == 1
        assert rows[0][0] is not None
        assert rows[0][1] == "failed"

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)

    def test_retry_increments_retry_count(self, db_session):
        """Assert retrying a capture increments the retry_count column."""
        httpx.post(f"{DEVICE_URL}/device/disconnect", timeout=5.0)

        resp = httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-SQL-003",
                "session_id": "SESS-SQL-003",
                "image_type": "ultrasound",
            },
            timeout=15.0,
        )
        capture_id = resp.json()["id"]

        httpx.post(f"{DEVICE_URL}/device/reconnect", timeout=5.0)
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)

        httpx.post(
            f"{BACKEND_URL}/api/captures/{capture_id}/retry", timeout=15.0
        )
        db_session.commit()

        rows = run_sql(
            db_session,
            "SELECT retry_count FROM captures WHERE id = :id",
            {"id": capture_id},
        )
        assert len(rows) == 1
        assert rows[0][0] >= 1

    def test_defect_submission_creates_row(self, db_session):
        """Assert defect submission creates a defect row with correct status."""
        before = count_rows(db_session, "defects")

        httpx.post(
            f"{BACKEND_URL}/api/defects",
            json={
                "title": "SQL validation defect test",
                "severity": "minor",
                "priority": "low",
            },
            timeout=10.0,
        )
        db_session.commit()
        after = count_rows(db_session, "defects")
        assert after == before + 1

        rows = run_sql(
            db_session,
            "SELECT status FROM defects WHERE title = :title",
            {"title": "SQL validation defect test"},
        )
        assert len(rows) >= 1
        assert rows[0][0] == "open"

    def test_dashboard_counts_match_database(self, db_session):
        """Assert dashboard summary counts match actual database state."""
        summary = httpx.get(
            f"{BACKEND_URL}/api/dashboard/summary", timeout=10.0
        ).json()

        db_total = count_rows(db_session, "captures")
        db_success = count_rows(
            db_session, "captures", "capture_status = 'success'"
        )
        db_failed = count_rows(
            db_session, "captures", "capture_status = 'failed'"
        )
        db_defects = count_rows(db_session, "defects")

        assert summary["total_captures"] == db_total
        assert summary["successful_captures"] == db_success
        assert summary["failed_captures"] == db_failed
        assert summary["total_defects"] == db_defects

    def test_device_events_logged(self, db_session):
        """Assert device events are written to the audit log table."""
        httpx.post(f"{DEVICE_URL}/device/reset", timeout=5.0)
        before = count_rows(db_session, "device_events")

        httpx.post(
            f"{BACKEND_URL}/api/captures",
            json={
                "patient_id": "PAT-SQL-EVT",
                "session_id": "SESS-SQL-EVT",
                "image_type": "fluoroscopy",
            },
            timeout=15.0,
        )
        db_session.commit()
        after = count_rows(db_session, "device_events")
        assert after > before
