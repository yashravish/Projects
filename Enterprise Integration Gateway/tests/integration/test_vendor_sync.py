"""
Integration tests for Vendor XML sync service.
"""
import pytest
from sqlalchemy import select

from app.models.failed_record import FailedRecord
from app.models.order import Order
from app.models.shipment import Shipment
from app.models.sync_job import SyncJob
from app.services.sync_service import execute_vendor_sync
from mock_providers.vendor.data import ORDERS_XML, SHIPMENTS_XML

CLEAN_ORDERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrderFeed>
    <Order>
        <OrderId>INT-ORD-5001</OrderId>
        <ExternalCustomerId>CRM-INT-001</ExternalCustomerId>
        <OrderDate>2024-01-12T08:30:00Z</OrderDate>
        <Status>confirmed</Status>
        <Currency>USD</Currency>
        <TotalAmount>500.00</TotalAmount>
    </Order>
</OrderFeed>
"""

CLEAN_SHIPMENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ShipmentFeed>
    <Shipment>
        <ShipmentId>INT-SHIP-6001</ShipmentId>
        <VendorOrderId>INT-ORD-5001</VendorOrderId>
        <TrackingNumber>TEST123456</TrackingNumber>
        <Carrier>UPS</Carrier>
        <Status>in_transit</Status>
        <EstimatedDelivery>2024-01-20T00:00:00Z</EstimatedDelivery>
        <WeightKg>2.500</WeightKg>
    </Shipment>
</ShipmentFeed>
"""


class TestVendorSyncService:
    def test_sync_inserts_orders(self, db, monkeypatch):
        from app.clients.vendor_client import VendorClient
        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: CLEAN_ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: "<ShipmentFeed/>")
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)

        result = execute_vendor_sync(db, triggered_by="test")

        orders = db.scalars(select(Order).where(Order.source == "vendor")).all()
        assert any(o.order_number == "INT-ORD-5001" for o in orders)
        assert result.records_inserted >= 1

    def test_sync_inserts_shipments(self, db, monkeypatch):
        from app.clients.vendor_client import VendorClient
        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: CLEAN_ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: CLEAN_SHIPMENTS_XML)
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)

        execute_vendor_sync(db, triggered_by="test")

        shipments = db.scalars(select(Shipment).where(Shipment.source == "vendor")).all()
        assert any(s.tracking_number == "TEST123456" for s in shipments)

    def test_malformed_order_goes_to_failed_records(self, db, monkeypatch):
        """The intentionally malformed record in ORDERS_XML should be captured."""
        from app.clients.vendor_client import VendorClient
        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: "<ShipmentFeed/>")
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)

        result = execute_vendor_sync(db, triggered_by="test")

        assert result.records_failed >= 1
        failed = db.scalars(
            select(FailedRecord).where(
                FailedRecord.sync_job_id == result.job_id,
                FailedRecord.source == "vendor",
            )
        ).all()
        assert len(failed) >= 1
        assert failed[0].record_type in ("order", "shipment")

    def test_creates_sync_job_record(self, db, monkeypatch):
        from app.clients.vendor_client import VendorClient
        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: "<OrderFeed/>")
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: "<ShipmentFeed/>")
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)

        result = execute_vendor_sync(db, triggered_by="test")

        job = db.get(SyncJob, result.job_id)
        assert job is not None
        assert job.job_type == "vendor_sync"

    def test_full_mock_data_sync(self, db, monkeypatch):
        """Smoke-test using the full realistic mock provider data."""
        from app.clients.vendor_client import VendorClient
        monkeypatch.setattr(VendorClient, "get_orders_xml", lambda self: ORDERS_XML)
        monkeypatch.setattr(VendorClient, "get_shipments_xml", lambda self: SHIPMENTS_XML)
        monkeypatch.setattr(VendorClient, "__enter__", lambda self: self)
        monkeypatch.setattr(VendorClient, "__exit__", lambda self, *a: None)

        result = execute_vendor_sync(db, triggered_by="test")

        # ORDERS_XML has 3 valid + 1 malformed = partial_success expected
        assert result.status in ("success", "partial_success")
        assert result.records_processed >= 4  # 3 valid orders + 1 malformed + 4 shipments
