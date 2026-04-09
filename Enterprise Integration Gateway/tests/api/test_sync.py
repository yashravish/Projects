"""
API tests for sync trigger endpoints.

Uses monkeypatching to replace the real HTTP clients with controlled responses,
so these tests run without a running mock_providers service.
"""
import pytest

from app.services import sync_service


class TestSyncEndpoints:
    def test_sync_crm_returns_200(self, client, monkeypatch):
        """CRM sync returns a valid SyncResult."""
        from app.clients.crm_client import CrmClient

        monkeypatch.setattr(
            CrmClient,
            "get_customers",
            lambda self: [
                {
                    "customerId": "TST-001",
                    "fullName": "Test Customer",
                    "emailAddress": "test@example.com",
                    "accountStatus": "active",
                    "billingAddress": {},
                }
            ],
        )
        monkeypatch.setattr(
            CrmClient,
            "get_orders",
            lambda self: [
                {
                    "orderId": "TST-ORD-001",
                    "customerId": "TST-001",
                    "orderNumber": "ORD-TST-001",
                    "status": "pending",
                    "totalAmount": 99.99,
                    "currency": "USD",
                }
            ],
        )
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)

        response = client.post("/api/v1/sync/crm")
        assert response.status_code == 200
        data = response.json()
        assert data["job_type"] == "crm_sync"
        assert data["status"] in ("success", "partial_success", "failed")
        assert "job_id" in data
        assert "correlation_id" in data

    def test_sync_vendor_returns_200(self, client, monkeypatch):
        """Vendor sync returns a valid SyncResult."""
        from app.clients.vendor_client import VendorClient
        from mock_providers.vendor.data import ORDERS_XML, SHIPMENTS_XML

        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: SHIPMENTS_XML)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)

        response = client.post("/api/v1/sync/vendor")
        assert response.status_code == 200
        data = response.json()
        assert data["job_type"] == "vendor_sync"
        assert data["status"] in ("success", "partial_success")

    def test_vendor_sync_captures_malformed_record(self, client, monkeypatch):
        """The malformed vendor order should appear in failed_records."""
        from app.clients.vendor_client import VendorClient
        from mock_providers.vendor.data import ORDERS_XML, SHIPMENTS_XML

        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: SHIPMENTS_XML)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)

        response = client.post("/api/v1/sync/vendor")
        assert response.status_code == 200
        data = response.json()
        # ORDERS_XML contains 1 malformed record
        assert data["records_failed"] >= 1

    def test_sync_all_returns_200(self, client, monkeypatch):
        """Full sync returns aggregated result."""
        from app.clients.crm_client import CrmClient
        from app.clients.vendor_client import VendorClient

        monkeypatch.setattr(CrmClient, "get_customers", lambda self: [])
        monkeypatch.setattr(CrmClient, "get_orders", lambda self: [])
        monkeypatch.setattr(CrmClient, "__exit__", lambda self, *a: None)
        monkeypatch.setattr(CrmClient, "__enter__", lambda self: self)

        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: "<OrderFeed/>")
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: "<ShipmentFeed/>")
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)

        response = client.post("/api/v1/sync/all")
        assert response.status_code == 200
        data = response.json()
        assert data["job_type"] == "full_sync"
