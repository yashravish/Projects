"""Schemas for opportunities."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from execsim.schemas.common import OpportunityTypeEnum, SideEnum


class OpportunitySummary(BaseModel):
    """Brief opportunity info for list endpoints."""
    id: UUID
    step: int = Field(ge=0)
    type: OpportunityTypeEnum
    side: SideEnum
    estimated_value_bps: float
    arrival_mid: float = Field(gt=0)

    model_config = {"from_attributes": True}


class OpportunityDetail(BaseModel):
    """Full opportunity detail."""
    id: UUID
    run_id: UUID
    step: int = Field(ge=0)
    type: OpportunityTypeEnum
    side: SideEnum
    estimated_value_bps: float
    arrival_mid: float = Field(gt=0)
    edge_bps: float
    detail: dict[str, Any]
    detected_at: datetime

    model_config = {"from_attributes": True}


class OpportunityListParams(BaseModel):
    """Query parameters for listing opportunities."""
    type: Optional[OpportunityTypeEnum] = None
    min_value_bps: Optional[float] = Field(default=None, ge=0)
