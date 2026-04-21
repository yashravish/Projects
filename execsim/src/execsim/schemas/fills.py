"""Schemas for fills."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from execsim.schemas.common import VenueEnum


class FillDetail(BaseModel):
    """Full fill detail."""
    id: UUID
    opportunity_id: UUID
    venue: VenueEnum
    requested_qty: float = Field(gt=0)
    filled_qty: float = Field(gt=0)
    exec_price: float = Field(gt=0)
    decision_price: float = Field(gt=0)
    arrival_mid: float = Field(gt=0)
    latency_steps: int = Field(ge=0)
    executed_at: datetime

    model_config = {"from_attributes": True}
