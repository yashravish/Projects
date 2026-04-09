from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.schemas.shipment import ShipmentResponse
from app.services.shipment_service import get_shipment_by_id, list_shipments

router = APIRouter()


@router.get("", response_model=list[ShipmentResponse], summary="List shipments")
def get_shipments(
    source: str | None = Query(None, description="Filter by source: 'vendor'"),
    status: str | None = Query(None, description="Filter by shipment status"),
    order_id: int | None = Query(None, description="Filter by internal order ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return paginated list of shipments synced from the vendor feed."""
    shipments, _ = list_shipments(
        db, source=source, status=status, order_id=order_id, skip=skip, limit=limit
    )
    return shipments


@router.get("/{shipment_id}", response_model=ShipmentResponse, summary="Get shipment by ID")
def get_shipment(shipment_id: int, db: Session = Depends(get_db)):
    """Return a single shipment by internal ID."""
    shipment = get_shipment_by_id(db, shipment_id)
    if shipment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Shipment {shipment_id} not found"
        )
    return shipment
