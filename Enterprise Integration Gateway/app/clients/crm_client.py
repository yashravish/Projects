"""
HTTP client for the mock CRM system.

Fetches customers and orders as JSON from the mock CRM service.
"""
import logging
from typing import Any

from app.clients.base_client import BaseHttpClient
from app.core.config import settings
from app.utils.json_parser import extract_list, safe_parse_json

logger = logging.getLogger(__name__)


class CrmClient(BaseHttpClient):
    """Client for the CRM mock API (JSON)."""

    def __init__(self) -> None:
        super().__init__(base_url=settings.CRM_BASE_URL, source_name="crm")

    def get_customers(self) -> list[dict[str, Any]]:
        """Fetch all customers from the CRM API."""
        response = self.get("/mock/crm/customers")
        payload = safe_parse_json(response.text, context="CRM customers")
        if payload is None:
            logger.error("crm_customers_parse_failed")
            return []
        records = extract_list(payload, "customers")
        logger.info("crm_customers_fetched", extra={"count": len(records)})
        return records

    def get_orders(self) -> list[dict[str, Any]]:
        """Fetch all orders from the CRM API."""
        response = self.get("/mock/crm/orders")
        payload = safe_parse_json(response.text, context="CRM orders")
        if payload is None:
            logger.error("crm_orders_parse_failed")
            return []
        records = extract_list(payload, "orders")
        logger.info("crm_orders_fetched", extra={"count": len(records)})
        return records
