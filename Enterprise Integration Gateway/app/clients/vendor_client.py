"""
HTTP client for the mock Vendor XML system.

Fetches orders and shipments as XML from the mock vendor service.
"""
import logging

from app.clients.base_client import BaseHttpClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class VendorClient(BaseHttpClient):
    """Client for the Vendor mock API (XML)."""

    def __init__(self) -> None:
        super().__init__(base_url=settings.VENDOR_BASE_URL, source_name="vendor")

    def get_orders_xml(self) -> str:
        """Fetch vendor orders as raw XML string."""
        response = self.get("/mock/vendor/orders")
        logger.info(
            "vendor_orders_fetched",
            extra={"content_length": len(response.text)},
        )
        return response.text

    def get_shipments_xml(self) -> str:
        """Fetch vendor shipments as raw XML string."""
        response = self.get("/mock/vendor/shipments")
        logger.info(
            "vendor_shipments_fetched",
            extra={"content_length": len(response.text)},
        )
        return response.text
