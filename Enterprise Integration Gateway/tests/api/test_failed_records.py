"""API tests for the /failed-records endpoints including retry logic."""
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from app.models.failed_record import FailedRecord
from app.models.sync_job import SyncJob


def _seed_job(db):
    job = SyncJob(
        correlation_id=str(uuid.uuid4()),
        job_type="crm_sync",
        status="partial_success",
        triggered_by="api",
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _seed_failed_record(
    db,
    job_id=None,
    source="crm",
    record_type="customer",
    retry_count=0,
    status="pending_retry",
    raw_data=None,
):
    raw = raw_data or '{"customerId": "TEST-001", "fullName": "Test User"}'
    record = FailedRecord(
        sync_job_id=job_id,
        source=source,
        record_type=record_type,
        external_id="TEST-EXT-001",
        raw_data=raw,
        error_message="Test failure reason",
        retry_count=retry_count,
        status=status,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


class TestListFailedRecords:
    def test_empty_returns_list(self, client):
        response = client.get("/api/v1/failed-records")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_seeded_record(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_failed_record(db)
        db.close()

        data = client.get("/api/v1/failed-records").json()
        assert len(data) >= 1

    def test_filter_by_source(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_failed_record(db, source="crm")
        _seed_failed_record(db, source="vendor", record_type="order")
        db.close()

        data = client.get("/api/v1/failed-records?source=vendor").json()
        assert all(r["source"] == "vendor" for r in data)

    def test_filter_by_status(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_failed_record(db, status="pending_retry")
        _seed_failed_record(db, source="vendor", status="resolved", record_type="order")
        db.close()

        data = client.get("/api/v1/failed-records?status=resolved").json()
        assert all(r["status"] == "resolved" for r in data)

    def test_filter_by_record_type(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_failed_record(db, record_type="customer")
        _seed_failed_record(db, source="vendor", record_type="shipment")
        db.close()

        data = client.get("/api/v1/failed-records?record_type=customer").json()
        assert all(r["record_type"] == "customer" for r in data)


class TestRetryFailedRecord:
    def test_successful_retry_resolves_crm_customer(self, client, engine):
        """
        A valid CRM customer raw payload should re-transform and resolve the record.
        """
        Session = sessionmaker(bind=engine)
        db = Session()
        record = _seed_failed_record(db, source="crm", record_type="customer")
        rid = record.id
        db.close()

        response = client.post(f"/api/v1/failed-records/{rid}/retry")
        assert response.status_code == 200
        data = response.json()
        assert data["record_id"] == rid
        assert data["status"] == "resolved"

    def test_nonexistent_record_returns_404(self, client):
        response = client.post("/api/v1/failed-records/99999/retry")
        assert response.status_code == 404

    def test_exhausted_retries_returns_409(self, client, engine):
        from app.core.config import settings
        Session = sessionmaker(bind=engine)
        db = Session()
        record = _seed_failed_record(db, retry_count=settings.MAX_RETRY_COUNT)
        rid = record.id
        db.close()

        response = client.post(f"/api/v1/failed-records/{rid}/retry")
        assert response.status_code == 409

    def test_retry_increments_count(self, client, engine):
        """After a failed retry the count should still increment."""
        Session = sessionmaker(bind=engine)
        db = Session()
        # Invalid transformation will fail (empty fullName / customerId)
        record = _seed_failed_record(
            db,
            source="crm",
            record_type="customer",
            raw_data='{"customerId": "", "fullName": ""}',
        )
        rid = record.id
        db.close()

        client.post(f"/api/v1/failed-records/{rid}/retry")

        # Verify retry_count incremented via another session
        Session2 = sessionmaker(bind=engine)
        db2 = Session2()
        refreshed = db2.get(FailedRecord, rid)
        assert refreshed.retry_count == 1
        db2.close()
