from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from app.schemas.common import OrmBase


class OrderBase(OrmBase):
    external_id: str = Field(..., description="Identifier from the source system")
    customer_id: int | None = None
    source: str = Field(..., description="Origin system: 'crm' or 'vendor'")
    order_number: str
    status: str = "pending"
    total_amount: Decimal | None = None
    currency: str = "USD"
    order_date: datetime | None = None
    notes: str | None = None
    raw_data: dict[str, Any] | None = None


class OrderCreate(OrderBase):
    pass


class OrderUpdate(OrmBase):
    status: str | None = None
    total_amount: Decimal | None = None
    notes: str | None = None


class OrderResponse(OrderBase):
    id: int
    created_at: datetime
    updated_at: datetime
