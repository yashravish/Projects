"""Rehearsal cycle schemas for request/response models.

Implements contracts for:
- POST /teams/{team_id}/cycles
- GET /teams/{team_id}/cycles
- GET /teams/{team_id}/cycles/active
"""

import uuid
from datetime import date as date_type
from datetime import datetime

from pydantic import BaseModel, Field


class CreateCycleRequest(BaseModel):
    """Request schema for creating a rehearsal cycle.

    Note: The model uses `name` for what the API exposes as `label`.
    """

    date: date_type = Field(description="Rehearsal date in YYYY-MM-DD format")
    label: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Optional label for the cycle. Auto-generated from date if not provided.",
    )


class CycleResponse(BaseModel):
    """Rehearsal cycle data in responses."""

    id: uuid.UUID
    team_id: uuid.UUID
    name: str  # This is the `label` from the API perspective
    date: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateCycleResponse(BaseModel):
    """Response schema for cycle creation."""

    cycle: CycleResponse


class CyclesListResponse(BaseModel):
    """Paginated list of rehearsal cycles.

    Follows pagination contract from systemprompt.md:
    - items: list of cycles
    - next_cursor: opaque string or null
    """

    items: list[CycleResponse]
    next_cursor: str | None


class ActiveCycleResponse(BaseModel):
    """Response schema for active cycle.

    Active cycle selection logic:
    1. Nearest upcoming cycle (date >= today) - sorted by date ASC
    2. Else, latest past cycle (date < today) - sorted by date DESC
    3. Else, null if no cycles exist
    """

    cycle: CycleResponse | None

