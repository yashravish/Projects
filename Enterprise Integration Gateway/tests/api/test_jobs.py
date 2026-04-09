"""API tests for the /integration-jobs endpoints."""
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from app.models.sync_job import SyncJob


def _seed_job(db, job_type="crm_sync", status="success"):
    job = SyncJob(
        correlation_id=str(uuid.uuid4()),
        job_type=job_type,
        status=status,
        triggered_by="api",
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
        completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        records_processed=10,
        records_inserted=8,
        records_updated=2,
        records_failed=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


class TestListIntegrationJobs:
    def test_empty_returns_list(self, client):
        response = client.get("/api/v1/integration-jobs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_seeded_job(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_job(db)
        db.close()

        data = client.get("/api/v1/integration-jobs").json()
        assert len(data) >= 1

    def test_filter_by_job_type(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_job(db, job_type="crm_sync")
        _seed_job(db, job_type="vendor_sync")
        db.close()

        data = client.get("/api/v1/integration-jobs?job_type=crm_sync").json()
        assert all(j["job_type"] == "crm_sync" for j in data)

    def test_filter_by_status(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_job(db, status="success")
        _seed_job(db, job_type="vendor_sync", status="failed")
        db.close()

        data = client.get("/api/v1/integration-jobs?status=failed").json()
        assert all(j["status"] == "failed" for j in data)


class TestGetIntegrationJob:
    def test_existing_job(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        job = _seed_job(db)
        jid = job.id
        db.close()

        data = client.get(f"/api/v1/integration-jobs/{jid}").json()
        assert data["id"] == jid
        assert data["job_type"] == "crm_sync"

    def test_nonexistent_returns_404(self, client):
        response = client.get("/api/v1/integration-jobs/99999")
        assert response.status_code == 404

    def test_job_contains_count_fields(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        job = _seed_job(db)
        jid = job.id
        db.close()

        data = client.get(f"/api/v1/integration-jobs/{jid}").json()
        assert data["records_processed"] == 10
        assert data["records_inserted"] == 8
        assert data["records_updated"] == 2
        assert data["records_failed"] == 0
