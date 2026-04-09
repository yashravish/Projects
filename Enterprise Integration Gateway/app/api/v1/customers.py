from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.core.exceptions import RecordNotFoundError
from app.schemas.customer import CustomerResponse
from app.services.customer_service import get_customer_by_id, list_customers

router = APIRouter()


@router.get("", response_model=list[CustomerResponse], summary="List customers")
def get_customers(
    source: str | None = Query(None, description="Filter by source: 'crm' or 'vendor'"),
    status: str | None = Query(None, description="Filter by status: 'active', 'inactive'"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Return paginated list of normalized customers from all integrated sources.
    """
    customers, _ = list_customers(db, source=source, status=status, skip=skip, limit=limit)
    return customers


@router.get("/{customer_id}", response_model=CustomerResponse, summary="Get customer by ID")
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    """Return a single customer by internal ID."""
    customer = get_customer_by_id(db, customer_id)
    if customer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")
    return customer
