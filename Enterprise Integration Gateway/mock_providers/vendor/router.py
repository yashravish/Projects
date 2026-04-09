"""
Mock Vendor API endpoints.

These simulate a legacy vendor EDI/XML feed system.
Returns XML responses with PascalCase tags.
"""
from fastapi import APIRouter
from fastapi.responses import Response

from mock_providers.vendor.data import ORDERS_XML, SHIPMENTS_XML

router = APIRouter()

_XML_CONTENT_TYPE = "application/xml; charset=utf-8"


@router.get("/orders", summary="[Mock Vendor] Get orders XML feed")
def get_orders():
    """Returns vendor orders as XML (includes one intentionally malformed record)."""
    return Response(content=ORDERS_XML, media_type=_XML_CONTENT_TYPE)


@router.get("/shipments", summary="[Mock Vendor] Get shipments XML feed")
def get_shipments():
    """Returns vendor shipments as XML."""
    return Response(content=SHIPMENTS_XML, media_type=_XML_CONTENT_TYPE)
