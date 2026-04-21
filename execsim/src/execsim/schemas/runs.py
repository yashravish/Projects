"""Schemas for simulation runs."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from execsim.schemas.common import RunStatusEnum


class RunCreate(BaseModel):
    """Request body to start a simulation run."""
    seed: int = Field(..., ge=0, description="Random seed for reproducibility.")
    num_steps: int = Field(default=1000, ge=1, description="Number of simulation steps.")
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for simulator config parameters.",
    )


class RunResponse(BaseModel):
    """Response after creating a run."""
    id: UUID
    seed: int
    status: RunStatusEnum
    started_at: datetime

    model_config = {"from_attributes": True}


class RunSummary(BaseModel):
    """Summary of a simulation run for list endpoints."""
    id: UUID
    seed: int
    status: RunStatusEnum
    num_steps: int
    started_at: datetime
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RunDetail(BaseModel):
    """Full detail of a simulation run."""
    id: UUID
    seed: int
    status: RunStatusEnum
    num_steps: int
    config: dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime] = None
    num_opportunities: int = Field(ge=0)
    num_fills: int = Field(ge=0)

    model_config = {"from_attributes": True}


class RunListParams(BaseModel):
    """Query parameters for listing runs."""
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    status: Optional[RunStatusEnum] = None
