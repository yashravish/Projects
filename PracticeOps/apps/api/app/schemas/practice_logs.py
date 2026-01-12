"""Practice log schemas for request/response models.

Implements contracts for:
- POST /cycles/{cycle_id}/practice-logs
- GET /cycles/{cycle_id}/practice-logs
- PATCH /practice-logs/{id}
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from app.models.enums import Priority, TicketCategory, TicketVisibility

if TYPE_CHECKING:
    from app.models import PracticeLog


class CreatePracticeLogRequest(BaseModel):
    """Request schema for creating a practice log.

    Validation rules:
    - occurred_at defaults to now
    - duration_min must be 1..600
    - rating_1_5 if provided must be 1..5
    - assignment_ids must exist and be visible to user
    """

    occurred_at: datetime | None = None
    duration_min: int = Field(ge=1, le=600)
    notes: str | None = Field(default=None, max_length=2000)
    rating_1_5: int | None = Field(default=None, ge=1, le=5)
    blocked_flag: bool = False
    assignment_ids: list[uuid.UUID] = Field(default_factory=list)


class SuggestedTicket(BaseModel):
    """Suggested ticket structure returned when blocked_flag=true.

    This is NOT ticket creation - just a structured suggestion.
    """

    title_suggestion: str
    due_date: str  # YYYY-MM-DD format
    visibility_default: TicketVisibility
    priority_default: Priority
    category_default: TicketCategory


class PracticeLogAssignmentResponse(BaseModel):
    """Assignment info in practice log response."""

    id: uuid.UUID
    title: str
    type: str


class PracticeLogResponse(BaseModel):
    """Practice log data in responses."""

    id: uuid.UUID
    user_id: uuid.UUID
    team_id: uuid.UUID
    cycle_id: uuid.UUID | None
    duration_minutes: int
    rating_1_5: int | None
    blocked_flag: bool
    notes: str | None
    occurred_at: datetime
    created_at: datetime
    assignments: list[PracticeLogAssignmentResponse]

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(
        cls, log: PracticeLog, assignments: list[PracticeLogAssignmentResponse] | None = None
    ) -> PracticeLogResponse:
        """Create response from model."""
        return cls(
            id=log.id,
            user_id=log.user_id,
            team_id=log.team_id,
            cycle_id=log.cycle_id,
            duration_minutes=log.duration_minutes,
            rating_1_5=log.rating_1_5,
            blocked_flag=log.blocked_flag,
            notes=log.notes,
            occurred_at=log.occurred_at,
            created_at=log.created_at,
            assignments=assignments or [],
        )


class CreatePracticeLogResponse(BaseModel):
    """Response schema for practice log creation."""

    practice_log: PracticeLogResponse
    suggested_ticket: SuggestedTicket | None = None


class PracticeLogsListResponse(BaseModel):
    """Paginated list of practice logs.

    Follows pagination contract from systemprompt.md:
    - items: list of practice logs
    - next_cursor: opaque string or null
    """

    items: list[PracticeLogResponse]
    next_cursor: str | None


class UpdatePracticeLogRequest(BaseModel):
    """Request schema for updating a practice log.

    Partial update - only provided fields are updated.
    Owner only - validated at route level.
    """

    occurred_at: datetime | None = None
    duration_min: int | None = Field(default=None, ge=1, le=600)
    notes: str | None = Field(default=None, max_length=2000)
    rating_1_5: int | None = Field(default=None, ge=1, le=5)
    blocked_flag: bool | None = None
    assignment_ids: list[uuid.UUID] | None = None

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> UpdatePracticeLogRequest:
        """Ensure at least one field is provided for update."""
        if all(
            v is None
            for v in [
                self.occurred_at,
                self.duration_min,
                self.notes,
                self.rating_1_5,
                self.blocked_flag,
                self.assignment_ids,
            ]
        ):
            raise ValueError("At least one field must be provided for update")
        return self


class UpdatePracticeLogResponse(BaseModel):
    """Response schema for practice log update."""

    practice_log: PracticeLogResponse

