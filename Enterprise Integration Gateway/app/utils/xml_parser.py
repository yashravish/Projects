"""
Utilities for safely parsing XML payloads from the vendor feed.

Uses the standard library xml.etree.ElementTree with defusedxml-style
precautions: entity expansion is not relevant for ElementTree by default,
but we avoid lxml's network-fetching and limit input size in the caller.
"""
import logging
from typing import Any
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, ParseError

logger = logging.getLogger(__name__)


def parse_xml_string(raw: str | bytes, context: str = "") -> Element | None:
    """
    Parse XML from a string, returning None on failure.

    Args:
        raw: Raw XML string or bytes.
        context: Human-readable label for logging.

    Returns:
        Root Element or None.
    """
    try:
        if isinstance(raw, bytes):
            return ET.fromstring(raw)
        return ET.fromstring(raw.encode("utf-8"))
    except ParseError as exc:
        logger.error(
            "xml_parse_error",
            extra={"context": context, "error": str(exc), "snippet": str(raw)[:300]},
        )
        return None


def elem_text(element: Element | None, tag: str, default: str | None = None) -> str | None:
    """Extract stripped text content of a child element, or *default* if missing/empty."""
    if element is None:
        return default
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    stripped = child.text.strip()
    return stripped if stripped else default


def elem_attrib(element: Element | None, tag: str, attrib: str, default: str | None = None) -> str | None:
    """Extract an attribute from a named child element."""
    if element is None:
        return default
    child = element.find(tag)
    if child is None:
        return default
    return child.get(attrib, default)


def parse_vendor_orders(xml_string: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse the vendor OrderFeed XML.

    Returns:
        (valid_orders, malformed_records) — each a list of dicts.
        Malformed records contain {'raw': str, 'error': str}.
    """
    root = parse_xml_string(xml_string, context="vendor_orders")
    if root is None:
        return [], [{"raw": xml_string, "error": "XML root could not be parsed"}]

    valid: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []

    for order_elem in root.findall("Order"):
        raw_str = ET.tostring(order_elem, encoding="unicode")
        try:
            order_id = elem_text(order_elem, "OrderId")
            if not order_id:
                raise ValueError("Missing or empty <OrderId>")

            total_amount_str = elem_text(order_elem, "TotalAmount")
            total_amount: float | None = None
            if total_amount_str:
                try:
                    total_amount = float(total_amount_str)
                except ValueError:
                    raise ValueError(f"Invalid <TotalAmount>: {total_amount_str!r}")

            record: dict[str, Any] = {
                "order_id": order_id,
                "external_customer_id": elem_text(order_elem, "ExternalCustomerId"),
                "order_date": elem_text(order_elem, "OrderDate"),
                "status": elem_text(order_elem, "Status") or "unknown",
                "currency": elem_text(order_elem, "Currency") or "USD",
                "total_amount": total_amount,
                "notes": elem_text(order_elem, "Notes"),
                "raw_xml": raw_str,
            }
            valid.append(record)
        except (ValueError, AttributeError) as exc:
            logger.warning(
                "vendor_order_malformed",
                extra={"error": str(exc), "raw": raw_str[:300]},
            )
            malformed.append({"raw": raw_str, "error": str(exc)})

    return valid, malformed


def parse_vendor_shipments(xml_string: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse the vendor ShipmentFeed XML.

    Returns:
        (valid_shipments, malformed_records)
    """
    root = parse_xml_string(xml_string, context="vendor_shipments")
    if root is None:
        return [], [{"raw": xml_string, "error": "XML root could not be parsed"}]

    valid: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []

    for ship_elem in root.findall("Shipment"):
        raw_str = ET.tostring(ship_elem, encoding="unicode")
        try:
            shipment_id = elem_text(ship_elem, "ShipmentId")
            if not shipment_id:
                raise ValueError("Missing or empty <ShipmentId>")

            weight_str = elem_text(ship_elem, "WeightKg")
            weight: float | None = None
            if weight_str:
                try:
                    weight = float(weight_str)
                except ValueError:
                    raise ValueError(f"Invalid <WeightKg>: {weight_str!r}")

            record: dict[str, Any] = {
                "shipment_id": shipment_id,
                "vendor_order_id": elem_text(ship_elem, "VendorOrderId"),
                "tracking_number": elem_text(ship_elem, "TrackingNumber"),
                "carrier": elem_text(ship_elem, "Carrier"),
                "status": elem_text(ship_elem, "Status") or "unknown",
                "estimated_delivery": elem_text(ship_elem, "EstimatedDelivery"),
                "actual_delivery": elem_text(ship_elem, "ActualDelivery"),
                "weight_kg": weight,
                "raw_xml": raw_str,
            }
            valid.append(record)
        except (ValueError, AttributeError) as exc:
            logger.warning(
                "vendor_shipment_malformed",
                extra={"error": str(exc), "raw": raw_str[:300]},
            )
            malformed.append({"raw": raw_str, "error": str(exc)})

    return valid, malformed
