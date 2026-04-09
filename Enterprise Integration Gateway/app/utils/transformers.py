"""
Data transformation layer.

Converts raw external payloads (CRM JSON dicts, Vendor XML dicts)
into normalized internal Pydantic schemas ready for persistence.

Each function raises TransformationError on unrecoverable issues.
"""
import json
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.exceptions import TransformationError
from app.schemas.customer import CustomerCreate
from app.schemas.order import OrderCreate
from app.schemas.shipment import ShipmentCreate

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_decimal(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        raise TransformationError("field", field, f"Cannot convert {value!r} to Decimal")


def _parse_datetime(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise TransformationError("field", field, f"Unrecognised datetime format: {value!r}")


# ── CRM JSON Transformers ──────────────────────────────────────────────────────


def transform_crm_customer(raw: dict[str, Any]) -> CustomerCreate:
    """
    Map a CRM API customer record to the internal CustomerCreate schema.

    CRM field names use camelCase / nested 'billingAddress' object.
    """
    external_id = raw.get("customerId") or raw.get("id")
    if not external_id:
        raise TransformationError("customer", None, "Missing 'customerId' in CRM record")

    name = raw.get("fullName") or raw.get("name")
    if not name:
        raise TransformationError("customer", str(external_id), "Missing 'fullName' in CRM record")

    billing = raw.get("billingAddress") or {}
    return CustomerCreate(
        external_id=str(external_id),
        source="crm",
        name=str(name),
        email=raw.get("emailAddress") or raw.get("email"),
        phone=raw.get("phoneNumber") or raw.get("phone"),
        company=raw.get("companyName") or raw.get("company"),
        address_line1=billing.get("street") or billing.get("line1"),
        address_line2=billing.get("line2"),
        city=billing.get("city"),
        state=billing.get("state"),
        country=billing.get("country"),
        postal_code=billing.get("zip") or billing.get("postalCode"),
        status=(raw.get("accountStatus") or raw.get("status") or "active").lower(),
        raw_data=raw,
    )


def transform_crm_order(raw: dict[str, Any], customer_id: int | None = None) -> OrderCreate:
    """Map a CRM API order record to the internal OrderCreate schema."""
    external_id = raw.get("orderId") or raw.get("id")
    if not external_id:
        raise TransformationError("order", None, "Missing 'orderId' in CRM record")

    order_number = raw.get("orderNumber") or raw.get("reference") or str(external_id)
    amount = _parse_decimal(raw.get("totalAmount") or raw.get("amount"), "totalAmount")
    order_date = _parse_datetime(raw.get("orderDate") or raw.get("createdAt"), "orderDate")

    return OrderCreate(
        external_id=str(external_id),
        customer_id=customer_id,
        source="crm",
        order_number=str(order_number),
        status=(raw.get("status") or "pending").lower(),
        total_amount=amount,
        currency=raw.get("currency") or "USD",
        order_date=order_date,
        notes=raw.get("notes"),
        raw_data=raw,
    )


# ── Vendor XML Transformers ────────────────────────────────────────────────────


def transform_vendor_order(parsed: dict[str, Any], customer_id: int | None = None) -> OrderCreate:
    """
    Map a parsed vendor XML order dict to the internal OrderCreate schema.

    'parsed' is the output of xml_parser.parse_vendor_orders() — already a dict.
    """
    external_id = parsed.get("order_id")
    if not external_id:
        raise TransformationError("order", None, "Missing 'order_id' in vendor XML record")

    amount = _parse_decimal(parsed.get("total_amount"), "total_amount")
    order_date = _parse_datetime(parsed.get("order_date"), "order_date")

    return OrderCreate(
        external_id=f"VND-{external_id}",
        customer_id=customer_id,
        source="vendor",
        order_number=str(external_id),
        status=(parsed.get("status") or "pending").lower(),
        total_amount=amount,
        currency=parsed.get("currency") or "USD",
        order_date=order_date,
        notes=parsed.get("notes"),
        raw_data={"source": "vendor_xml", "raw_xml": parsed.get("raw_xml")},
    )


def transform_vendor_shipment(parsed: dict[str, Any], order_id: int | None = None) -> ShipmentCreate:
    """Map a parsed vendor XML shipment dict to the internal ShipmentCreate schema."""
    external_id = parsed.get("shipment_id")
    if not external_id:
        raise TransformationError("shipment", None, "Missing 'shipment_id' in vendor XML record")

    est_delivery = _parse_datetime(parsed.get("estimated_delivery"), "estimated_delivery")
    act_delivery = _parse_datetime(parsed.get("actual_delivery"), "actual_delivery")
    weight = _parse_decimal(parsed.get("weight_kg"), "weight_kg")

    return ShipmentCreate(
        external_id=f"VND-{external_id}",
        order_id=order_id,
        source="vendor",
        tracking_number=parsed.get("tracking_number"),
        carrier=parsed.get("carrier"),
        status=(parsed.get("status") or "pending").lower(),
        estimated_delivery=est_delivery,
        actual_delivery=act_delivery,
        weight_kg=weight,
        raw_data={"source": "vendor_xml", "raw_xml": parsed.get("raw_xml")},
    )
