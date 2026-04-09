"""
Unit tests for the XML parsing utilities.
"""
import pytest

from app.utils.xml_parser import (
    elem_text,
    parse_vendor_orders,
    parse_vendor_shipments,
    parse_xml_string,
)

VALID_ORDERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrderFeed>
    <Order>
        <OrderId>VND-ORD-2001</OrderId>
        <ExternalCustomerId>CRM-CUST-001</ExternalCustomerId>
        <OrderDate>2024-01-12T08:30:00Z</OrderDate>
        <Status>confirmed</Status>
        <Currency>USD</Currency>
        <TotalAmount>645.00</TotalAmount>
        <Notes>Test note</Notes>
    </Order>
</OrderFeed>
"""

MALFORMED_ORDERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrderFeed>
    <Order>
        <OrderId></OrderId>
        <TotalAmount>NOT_A_NUMBER</TotalAmount>
    </Order>
</OrderFeed>
"""

MIXED_ORDERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrderFeed>
    <Order>
        <OrderId>GOOD-001</OrderId>
        <Status>confirmed</Status>
        <TotalAmount>100.00</TotalAmount>
    </Order>
    <Order>
        <OrderId></OrderId>
        <TotalAmount>INVALID</TotalAmount>
    </Order>
</OrderFeed>
"""

VALID_SHIPMENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ShipmentFeed>
    <Shipment>
        <ShipmentId>VND-SHIP-3001</ShipmentId>
        <VendorOrderId>VND-ORD-2001</VendorOrderId>
        <TrackingNumber>1Z999AA10123456784</TrackingNumber>
        <Carrier>UPS</Carrier>
        <Status>delivered</Status>
        <EstimatedDelivery>2024-01-18T00:00:00Z</EstimatedDelivery>
        <WeightKg>3.200</WeightKg>
    </Shipment>
</ShipmentFeed>
"""


class TestParseXmlString:
    def test_valid_xml_returns_element(self):
        root = parse_xml_string("<root><child>text</child></root>")
        assert root is not None
        assert root.tag == "root"

    def test_invalid_xml_returns_none(self):
        result = parse_xml_string("<<not xml>>")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_xml_string("")
        assert result is None

    def test_bytes_input(self):
        root = parse_xml_string(b"<root/>")
        assert root is not None


class TestElemText:
    def test_existing_tag(self):
        root = parse_xml_string("<parent><name>Alice</name></parent>")
        assert elem_text(root, "name") == "Alice"

    def test_missing_tag_returns_default(self):
        root = parse_xml_string("<parent/>")
        assert elem_text(root, "missing") is None
        assert elem_text(root, "missing", default="N/A") == "N/A"

    def test_empty_tag_returns_default(self):
        root = parse_xml_string("<parent><name></name></parent>")
        assert elem_text(root, "name") is None

    def test_none_element(self):
        assert elem_text(None, "any") is None


class TestParseVendorOrders:
    def test_valid_orders_parsed(self):
        valid, malformed = parse_vendor_orders(VALID_ORDERS_XML)
        assert len(valid) == 1
        assert len(malformed) == 0
        order = valid[0]
        assert order["order_id"] == "VND-ORD-2001"
        assert order["status"] == "confirmed"
        assert order["total_amount"] == 645.00
        assert order["currency"] == "USD"

    def test_malformed_order_goes_to_failures(self):
        valid, malformed = parse_vendor_orders(MALFORMED_ORDERS_XML)
        assert len(valid) == 0
        assert len(malformed) == 1
        assert "error" in malformed[0]

    def test_mixed_orders_segregated(self):
        valid, malformed = parse_vendor_orders(MIXED_ORDERS_XML)
        assert len(valid) == 1
        assert len(malformed) == 1
        assert valid[0]["order_id"] == "GOOD-001"

    def test_completely_invalid_xml(self):
        valid, malformed = parse_vendor_orders("<<garbage>>")
        assert len(valid) == 0
        assert len(malformed) == 1

    def test_raw_xml_preserved_in_record(self):
        valid, _ = parse_vendor_orders(VALID_ORDERS_XML)
        assert "raw_xml" in valid[0]
        assert "VND-ORD-2001" in valid[0]["raw_xml"]


class TestParseVendorShipments:
    def test_valid_shipment(self):
        valid, malformed = parse_vendor_shipments(VALID_SHIPMENTS_XML)
        assert len(valid) == 1
        assert len(malformed) == 0
        ship = valid[0]
        assert ship["shipment_id"] == "VND-SHIP-3001"
        assert ship["carrier"] == "UPS"
        assert ship["weight_kg"] == 3.2

    def test_missing_shipment_id_fails(self):
        xml = "<ShipmentFeed><Shipment><ShipmentId></ShipmentId></Shipment></ShipmentFeed>"
        valid, malformed = parse_vendor_shipments(xml)
        assert len(valid) == 0
        assert len(malformed) == 1
