"""
CRUD + upsert operations for the Order entity.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.order import Order
from app.schemas.order import OrderCreate

logger = logging.getLogger(__name__)


def get_order_by_id(db: Session, order_id: int) -> Order | None:
    return db.get(Order, order_id)


def get_order_by_external_id(db: Session, external_id: str) -> Order | None:
    stmt = select(Order).where(Order.external_id == external_id)
    return db.scalars(stmt).first()


def list_orders(
    db: Session,
    source: str | None = None,
    status: str | None = None,
    customer_id: int | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[Order], int]:
    stmt = select(Order)
    if source:
        stmt = stmt.where(Order.source == source)
    if status:
        stmt = stmt.where(Order.status == status)
    if customer_id:
        stmt = stmt.where(Order.customer_id == customer_id)
    all_rows = db.scalars(stmt).all()
    return list(all_rows[skip: skip + limit]), len(all_rows)


def upsert_order(db: Session, data: OrderCreate) -> tuple[Order, bool]:
    """
    Insert or update an order keyed on external_id.

    Returns (Order, created).
    """
    existing = get_order_by_external_id(db, data.external_id)
    if existing:
        for field, value in data.model_dump(exclude={"external_id", "source"}).items():
            if value is not None:
                setattr(existing, field, value)
        existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.flush()
        logger.debug("order_updated", extra={"external_id": data.external_id})
        return existing, False

    order = Order(**data.model_dump())
    db.add(order)
    db.flush()
    logger.debug("order_inserted", extra={"external_id": data.external_id})
    return order, True
