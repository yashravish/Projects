"""
CRUD + upsert operations for the Customer entity.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.schemas.customer import CustomerCreate

logger = logging.getLogger(__name__)


def get_customer_by_id(db: Session, customer_id: int) -> Customer | None:
    return db.get(Customer, customer_id)


def get_customer_by_external_id(db: Session, external_id: str) -> Customer | None:
    stmt = select(Customer).where(Customer.external_id == external_id)
    return db.scalars(stmt).first()


def list_customers(
    db: Session,
    source: str | None = None,
    status: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[Customer], int]:
    stmt = select(Customer)
    if source:
        stmt = stmt.where(Customer.source == source)
    if status:
        stmt = stmt.where(Customer.status == status)
    total = db.scalars(stmt).all()
    paged = total[skip: skip + limit]
    return list(paged), len(total)


def upsert_customer(db: Session, data: CustomerCreate) -> tuple[Customer, bool]:
    """
    Insert or update a customer keyed on external_id.

    Returns:
        (Customer, created) — created is True if a new row was inserted.
    """
    existing = get_customer_by_external_id(db, data.external_id)
    if existing:
        for field, value in data.model_dump(exclude={"external_id", "source"}).items():
            if value is not None:
                setattr(existing, field, value)
        existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.flush()
        logger.debug("customer_updated", extra={"external_id": data.external_id})
        return existing, False

    customer = Customer(**data.model_dump())
    db.add(customer)
    db.flush()
    logger.debug("customer_inserted", extra={"external_id": data.external_id})
    return customer, True
