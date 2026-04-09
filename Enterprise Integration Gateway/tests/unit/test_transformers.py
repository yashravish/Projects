"""
Unit tests for the data transformation layer.

Verifies that CRM JSON and Vendor XML dictionaries are correctly mapped
into the internal Pydantic schemas.
"""
import pytest
from decimal import Decimal

from app.core.exceptions import TransformationError
from app.utils.transformers import (
    transform_crm_customer,
    transform_crm_order,
    transform_vendor_order,
    transform_vendor_shipment,
)


# ── CRM Customer ──────────────────────────────────────────────────────────────


class TestTransformCrmCustomer:
    def test_full_record(self):
        raw = {
            "customerId": "CRM-CUST-001",
            "fullName": "Alice Johnson",
            "emailAddress": "alice@example.com",
            "phoneNumber": "+1-555-0101",
            "companyName": "Acme Corp",
            "accountStatus": "active",
            "billingAddress": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "country": "US",
                "zip": "10001",
            },
        }
        result = transform_crm_customer(raw)
        assert result.external_id == "CRM-CUST-001"
        assert result.name == "Alice Johnson"
        assert result.email == "alice@example.com"
        assert result.source == "crm"
        assert result.status == "active"
        assert result.address_line1 == "123 Main St"
        assert result.city == "New York"
        assert result.postal_code == "10001"

    def test_missing_customer_id_raises(self):
        with pytest.raises(TransformationError):
            transform_crm_customer({"fullName": "No ID"})

    def test_missing_full_name_raises(self):
        with pytest.raises(TransformationError):
            transform_crm_customer({"customerId": "CRM-001"})

    def test_status_normalized_to_lowercase(self):
        raw = {"customerId": "X", "fullName": "Y", "accountStatus": "ACTIVE"}
        result = transform_crm_customer(raw)
        assert result.status == "active"

    def test_missing_billing_address_ok(self):
        raw = {"customerId": "CRM-002", "fullName": "Bob"}
        result = transform_crm_customer(raw)
        assert result.address_line1 is None
        assert result.city is None


# ── CRM Order ─────────────────────────────────────────────────────────────────


class TestTransformCrmOrder:
    def test_full_record(self):
        raw = {
            "orderId": "CRM-ORD-1001",
            "customerId": "CRM-CUST-001",
            "orderNumber": "ORD-2024-1001",
            "status": "shipped",
            "totalAmount": 1250.00,
            "currency": "USD",
            "orderDate": "2024-01-10T10:00:00Z",
        }
        result = transform_crm_order(raw, customer_id=42)
        assert result.external_id == "CRM-ORD-1001"
        assert result.order_number == "ORD-2024-1001"
        assert result.status == "shipped"
        assert result.total_amount == Decimal("1250.0")
        assert result.customer_id == 42
        assert result.source == "crm"

    def test_missing_order_id_raises(self):
        with pytest.raises(TransformationError):
            transform_crm_order({"orderNumber": "X"})

    def test_status_defaults_to_pending(self):
        raw = {"orderId": "CRM-ORD-X"}
        result = transform_crm_order(raw)
        assert result.status == "pending"

    def test_no_customer_id_ok(self):
        raw = {"orderId": "CRM-ORD-Y", "orderNumber": "ORD-Y"}
        result = transform_crm_order(raw)
        assert result.customer_id is None


# ── Vendor Order ──────────────────────────────────────────────────────────────


class TestTransformVendorOrder:
    def test_valid_record(self):
        parsed = {
            "order_id": "VND-ORD-2001",
            "external_customer_id": "CRM-CUST-001",
            "order_date": "2024-01-12T08:30:00Z",
            "status": "confirmed",
            "currency": "USD",
            "total_amount": 645.00,
            "raw_xml": "<Order>...</Order>",
        }
        result = transform_vendor_order(parsed, customer_id=1)
        assert result.external_id == "VND-VND-ORD-2001"
        assert result.source == "vendor"
        assert result.total_amount == Decimal("645.0")
        assert result.customer_id == 1

    def test_missing_order_id_raises(self):
        with pytest.raises(TransformationError):
            transform_vendor_order({"status": "ok"})

    def test_none_total_amount_ok(self):
        parsed = {"order_id": "VND-ORD-X", "total_amount": None}
        result = transform_vendor_order(parsed)
        assert result.total_amount is None


# ── Vendor Shipment ───────────────────────────────────────────────────────────


class TestTransformVendorShipment:
    def test_valid_record(self):
        parsed = {
            "shipment_id": "VND-SHIP-3001",
            "vendor_order_id": "VND-ORD-2001",
            "tracking_number": "1Z999AA10123456784",
            "carrier": "UPS",
            "status": "delivered",
            "estimated_delivery": "2024-01-18T00:00:00Z",
            "actual_delivery": "2024-01-17T14:32:00Z",
            "weight_kg": 3.2,
            "raw_xml": "<Shipment>...</Shipment>",
        }
        result = transform_vendor_shipment(parsed, order_id=10)
        assert result.external_id == "VND-VND-SHIP-3001"
        assert result.source == "vendor"
        assert result.tracking_number == "1Z999AA10123456784"
        assert result.carrier == "UPS"
        assert result.weight_kg == Decimal("3.2")
        assert result.order_id == 10

    def test_missing_shipment_id_raises(self):
        with pytest.raises(TransformationError):
            transform_vendor_shipment({"carrier": "UPS"})
