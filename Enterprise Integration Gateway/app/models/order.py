from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    customer_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("customers.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    order_number: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    total_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    order_date: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    raw_data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    customer: Mapped["Customer"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Customer", back_populates="orders", lazy="select"
    )
    shipments: Mapped[list["Shipment"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Shipment", back_populates="order", lazy="select"
    )

    __table_args__ = (
        Index("ix_orders_source_status", "source", "status"),
        Index("ix_orders_customer_id", "customer_id"),
    )

    def __repr__(self) -> str:
        return f"<Order id={self.id} external_id={self.external_id!r} status={self.status!r}>"
