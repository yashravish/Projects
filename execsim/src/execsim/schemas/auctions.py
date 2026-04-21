"""Schemas for auctions and calibration."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AuctionCreate(BaseModel):
    """Request body to run an auction on a completed run."""
    reserve_price_bps: float = Field(default=0.0, ge=0)
    n_bidders: int = Field(default=3, ge=1, le=100)


class AuctionResponse(BaseModel):
    """Response after running an auction."""
    id: UUID
    run_id: UUID
    reserve_price_bps: float = Field(ge=0)
    num_opportunities: int = Field(ge=0)
    num_allocated: int = Field(ge=0)
    created_at: datetime

    model_config = {"from_attributes": True}


class AuctionEntrySchema(BaseModel):
    """Single bid entry in an auction."""
    id: UUID
    opportunity_id: UUID
    bidder_index: int = Field(ge=0)
    bid_value_bps: float
    won: bool
    payment_bps: float = Field(ge=0)

    model_config = {"from_attributes": True}


class AuctionResultSchema(BaseModel):
    """Aggregate auction result."""
    total_revenue_bps: float
    allocation_rate: float = Field(ge=0, le=1)
    mean_payment_bps: float

    model_config = {"from_attributes": True}


class AuctionDetail(BaseModel):
    """Full auction detail with entries and result."""
    id: UUID
    run_id: UUID
    reserve_price_bps: float
    num_opportunities: int
    num_allocated: int
    created_at: datetime
    entries: list[AuctionEntrySchema]
    result: Optional[AuctionResultSchema] = None

    model_config = {"from_attributes": True}


class CalibrationRequest(BaseModel):
    """Request body for reserve-price calibration.

    The objective is to maximize expected auctioneer revenue on the held-out
    seeds, subject to allocation_rate >= allocation_floor.
    """
    training_seeds: list[int] = Field(
        ..., min_length=1, description="Seeds for training simulation runs.",
    )
    held_out_seeds: list[int] = Field(
        ..., min_length=1, description="Seeds for held-out evaluation runs.",
    )
    n_bidders: int = Field(default=3, ge=1)
    grid_max_bps: int = Field(default=50, ge=1)
    grid_step_bps: int = Field(default=1, ge=1)
    allocation_floor: float = Field(default=0.5, ge=0, le=1)


class CalibrationGridPoint(BaseModel):
    """Result for a single reserve-price candidate."""
    reserve_bps: float = Field(ge=0)
    mean_revenue_bps: float
    mean_allocation_rate: float = Field(ge=0, le=1)
    feasible: bool


class CalibrationResponse(BaseModel):
    """Response from reserve-price calibration."""
    optimal_reserve_bps: float = Field(ge=0)
    optimal_revenue_bps: float
    optimal_allocation_rate: float = Field(ge=0, le=1)
    grid: list[CalibrationGridPoint]
