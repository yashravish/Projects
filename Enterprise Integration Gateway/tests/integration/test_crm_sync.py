"""
Integration tests for CRM sync service (with mocked HTTP client).

These tests exercise the full sync_service pipeline including
transformation, upsert, and failed_record tracking — but mock the HTTP call.
"""
import pytest
from sqlalchemy import select

from app.models.customer import Customer
from app.models.failed_record import FailedRecord
from app.models.order import Order
from app.models.sync_job import SyncJob
from app.services.sync_service import execute_crm_sync


VALID_CUSTOMERS = [
    {
        "customerId": "CRM-INT-001",
        "fullName": "Integration Test User",
        "emailAddress": "int@test.com",
        "accountStatus": "active",
        "billingAddress": {
            "street": "1 Test St",
            "city": "TestCity",
            "state": "TS",
            "country": "US",
            "zip": "00001",
        },
    }
]

VALID_ORDERS = [
    {
        "orderId": "CRM-INT-ORD-001",
        "customerId": "CRM-INT-001",
        "orderNumber": "ORD-INT-001",
        "status": "pending",
        "totalAmount": 250.00,
        "currency": "USD",
        "orderDate": "2024-03-01T10:00:00Z",
    }
]

INVALID_CUSTOMER = [
    {
        "customerId": "",  # empty ID triggers TransformationError
        "fullName": "Bad Record",
    }
]


class TestCrmSyncService:
    def test_sync_inserts_customers(self, db, monkeypatch):
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: VALID_CUSTOMERS)
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: [])
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        result = execute_crm_sync(db, triggered_by="test")

        assert result.records_inserted >= 1
        customers = db.scalars(select(Customer)).all()
        assert any(c.external_id == "CRM-INT-001" for c in customers)

    def test_sync_inserts_orders(self, db, monkeypatch):
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: VALID_CUSTOMERS)
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: VALID_ORDERS)
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        result = execute_crm_sync(db, triggered_by="test")

        orders = db.scalars(select(Order)).all()
        assert any(o.external_id == "CRM-INT-ORD-001" for o in orders)

    def test_sync_links_order_to_customer(self, db, monkeypatch):
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: VALID_CUSTOMERS)
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: VALID_ORDERS)
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        execute_crm_sync(db, triggered_by="test")

        customer = db.scalars(
            select(Customer).where(Customer.external_id == "CRM-INT-001")
        ).first()
        order = db.scalars(
            select(Order).where(Order.external_id == "CRM-INT-ORD-001")
        ).first()
        assert order is not None
        assert order.customer_id == customer.id

    def test_sync_creates_sync_job_record(self, db, monkeypatch):
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: [])
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: [])
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        result = execute_crm_sync(db, triggered_by="test")

        job = db.get(SyncJob, result.job_id)
        assert job is not None
        assert job.job_type == "crm_sync"
        assert job.status in ("success", "partial_success", "failed")
        assert job.correlation_id == result.correlation_id

    def test_invalid_customer_goes_to_failed_records(self, db, monkeypatch):
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: INVALID_CUSTOMER)
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: [])
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        result = execute_crm_sync(db, triggered_by="test")

        assert result.records_failed >= 1
        failed = db.scalars(
            select(FailedRecord).where(FailedRecord.sync_job_id == result.job_id)
        ).all()
        assert len(failed) >= 1
        assert failed[0].source == "crm"
        assert failed[0].record_type == "customer"

    def test_upsert_updates_existing_customer(self, db, monkeypatch):
        """Running the same sync twice should update, not insert, on second run."""
        from app.clients.crm_client import CrmClient
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: VALID_CUSTOMERS)
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: [])
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)

        # First run
        execute_crm_sync(db, triggered_by="test")

        # Second run with updated name
        updated = [{**VALID_CUSTOMERS[0], "fullName": "Updated Name"}]
        monkeypatch.setattr(CrmClient, "get_customers", lambda self: updated)
        execute_crm_sync(db, triggered_by="test")

        customers = db.scalars(
            select(Customer).where(Customer.external_id == "CRM-INT-001")
        ).all()
        assert len(customers) == 1  # no duplicate
        assert customers[0].name == "Updated Name"
