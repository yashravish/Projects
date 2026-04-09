from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    order_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("orders.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    tracking_number: Mapped[str | None] = mapped_column(String(100), index=True)
    carrier: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    estimated_delivery: Mapped[datetime | None] = mapped_column(DateTime)
    actual_delivery: Mapped[datetime | None] = mapped_column(DateTime)
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(8, 3))
    raw_data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    order: Mapped["Order"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Order", back_populates="shipments", lazy="select"
    )

    __table_args__ = (
        Index("ix_shipments_source_status", "source", "status"),
        Index("ix_shipments_order_id", "order_id"),
    )

    def __repr__(self) -> str:
        return f"<Shipment id={self.id} external_id={self.external_id!r} status={self.status!r}>"
