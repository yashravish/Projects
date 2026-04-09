"""
CRUD + upsert operations for the Shipment entity.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.shipment import Shipment
from app.schemas.shipment import ShipmentCreate

logger = logging.getLogger(__name__)


def get_shipment_by_id(db: Session, shipment_id: int) -> Shipment | None:
    return db.get(Shipment, shipment_id)


def get_shipment_by_external_id(db: Session, external_id: str) -> Shipment | None:
    stmt = select(Shipment).where(Shipment.external_id == external_id)
    return db.scalars(stmt).first()


def list_shipments(
    db: Session,
    source: str | None = None,
    status: str | None = None,
    order_id: int | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[Shipment], int]:
    stmt = select(Shipment)
    if source:
        stmt = stmt.where(Shipment.source == source)
    if status:
        stmt = stmt.where(Shipment.status == status)
    if order_id:
        stmt = stmt.where(Shipment.order_id == order_id)
    all_rows = db.scalars(stmt).all()
    return list(all_rows[skip: skip + limit]), len(all_rows)


def upsert_shipment(db: Session, data: ShipmentCreate) -> tuple[Shipment, bool]:
    """
    Insert or update a shipment keyed on external_id.

    Returns (Shipment, created).
    """
    existing = get_shipment_by_external_id(db, data.external_id)
    if existing:
        for field, value in data.model_dump(exclude={"external_id", "source"}).items():
            if value is not None:
                setattr(existing, field, value)
        existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.flush()
        logger.debug("shipment_updated", extra={"external_id": data.external_id})
        return existing, False

    shipment = Shipment(**data.model_dump())
    db.add(shipment)
    db.flush()
    logger.debug("shipment_inserted", extra={"external_id": data.external_id})
    return shipment, True
