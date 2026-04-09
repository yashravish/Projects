from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.schemas.order import OrderResponse
from app.services.order_service import get_order_by_id, list_orders

router = APIRouter()


@router.get("", response_model=list[OrderResponse], summary="List orders")
def get_orders(
    source: str | None = Query(None, description="Filter by source: 'crm' or 'vendor'"),
    status: str | None = Query(None, description="Filter by order status"),
    customer_id: int | None = Query(None, description="Filter by internal customer ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return paginated list of normalized orders from all integrated sources."""
    orders, _ = list_orders(
        db, source=source, status=status, customer_id=customer_id, skip=skip, limit=limit
    )
    return orders


@router.get("/{order_id}", response_model=OrderResponse, summary="Get order by ID")
def get_order(order_id: int, db: Session = Depends(get_db)):
    """Return a single order by internal ID."""
    order = get_order_by_id(db, order_id)
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Order {order_id} not found")
    return order
