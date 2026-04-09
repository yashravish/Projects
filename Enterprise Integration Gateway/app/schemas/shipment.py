from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from app.schemas.common import OrmBase


class ShipmentBase(OrmBase):
    external_id: str = Field(..., description="Identifier from the source system")
    order_id: int | None = None
    source: str
    tracking_number: str | None = None
    carrier: str | None = None
    status: str = "pending"
    estimated_delivery: datetime | None = None
    actual_delivery: datetime | None = None
    weight_kg: Decimal | None = None
    raw_data: dict[str, Any] | None = None


class ShipmentCreate(ShipmentBase):
    pass


class ShipmentUpdate(OrmBase):
    status: str | None = None
    tracking_number: str | None = None
    actual_delivery: datetime | None = None


class ShipmentResponse(ShipmentBase):
    id: int
    created_at: datetime
    updated_at: datetime
