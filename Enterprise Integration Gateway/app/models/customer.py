from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # 'crm', 'vendor'
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), index=True)
    phone: Mapped[str | None] = mapped_column(String(50))
    company: Mapped[str | None] = mapped_column(String(255))
    address_line1: Mapped[str | None] = mapped_column(String(255))
    address_line2: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    postal_code: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)
    raw_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # original payload for audit
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    orders: Mapped[list["Order"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Order", back_populates="customer", lazy="select"
    )

    __table_args__ = (
        Index("ix_customers_source_status", "source", "status"),
    )

    def __repr__(self) -> str:
        return f"<Customer id={self.id} external_id={self.external_id!r} source={self.source!r}>"
