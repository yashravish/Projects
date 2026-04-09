"""
Mock CRM API endpoints.

These simulate an external CRM SaaS system returning JSON.
Deliberately uses camelCase and nested structures to require transformation.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mock_providers.crm.data import CUSTOMERS, ORDERS

router = APIRouter()


@router.get("/customers", summary="[Mock CRM] List customers")
def list_customers():
    """Returns all CRM customers as JSON (simulates external CRM pagination disabled for simplicity)."""
    return JSONResponse(content={"customers": CUSTOMERS, "total": len(CUSTOMERS)})


@router.get("/customers/{customer_id}", summary="[Mock CRM] Get customer")
def get_customer(customer_id: str):
    customer = next((c for c in CUSTOMERS if c["customerId"] == customer_id), None)
    if customer is None:
        return JSONResponse(status_code=404, content={"error": "Customer not found"})
    return JSONResponse(content=customer)


@router.get("/orders", summary="[Mock CRM] List orders")
def list_orders():
    """Returns all CRM orders as JSON."""
    return JSONResponse(content={"orders": ORDERS, "total": len(ORDERS)})


@router.get("/orders/{order_id}", summary="[Mock CRM] Get order")
def get_order(order_id: str):
    order = next((o for o in ORDERS if o["orderId"] == order_id), None)
    if order is None:
        return JSONResponse(status_code=404, content={"error": "Order not found"})
    return JSONResponse(content=order)
