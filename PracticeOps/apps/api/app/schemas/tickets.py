"""Ticket schemas for request/response models.

Implements contracts for:
- POST /cycles/{cycle_id}/tickets
- GET /cycles/{cycle_id}/tickets
- POST /tickets/{id}/claim
- PATCH /tickets/{id}
- GET /tickets/{id}/activity
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    Priority,
    TicketActivityType,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
)

if TYPE_CHECKING:
    from app.models import Ticket, TicketActivity


class CreateTicketRequest(BaseModel):
    """Request schema for creating a ticket.

    Rules:
    - visibility=SECTION requires section field
    - claimable=true requires owner_id=null (or not provided)
    - Server sets: due_date, status=OPEN, created_by
    """

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    category: TicketCategory
    song_ref: str | None = Field(default=None, max_length=100)
    priority: Priority
    visibility: TicketVisibility
    section: str | None = Field(default=None, max_length=50)
    owner_id: uuid.UUID | None = None
    claimable: bool = False

    @model_validator(mode="after")
    def validate_visibility_section(self) -> CreateTicketRequest:
        """Validate that SECTION visibility requires section field."""
        if self.visibility == TicketVisibility.SECTION and self.section is None:
            raise ValueError("SECTION visibility requires a section to be specified")
        return self

    @model_validator(mode="after")
    def validate_claimable_owner(self) -> CreateTicketRequest:
        """Validate that claimable tickets have no owner."""
        if self.claimable and self.owner_id is not None:
            raise ValueError("Claimable tickets cannot have an owner_id specified")
        return self


class TicketResponse(BaseModel):
    """Ticket data in responses."""

    id: uuid.UUID
    team_id: uuid.UUID
    cycle_id: uuid.UUID | None
    owner_id: uuid.UUID | None
    created_by: uuid.UUID
    claimed_by: uuid.UUID | None
    claimable: bool
    category: TicketCategory
    priority: Priority
    status: TicketStatus
    visibility: TicketVisibility
    section: str | None
    title: str
    description: str | None
    song_ref: str | None
    due_at: datetime | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    resolved_note: str | None
    verified_at: datetime | None
    verified_by: uuid.UUID | None
    verified_note: str | None

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(cls, ticket: Ticket) -> TicketResponse:
        """Create response from model."""
        return cls(
            id=ticket.id,
            team_id=ticket.team_id,
            cycle_id=ticket.cycle_id,
            owner_id=ticket.owner_id,
            created_by=ticket.created_by,
            claimed_by=ticket.claimed_by,
            claimable=ticket.claimable,
            category=ticket.category,
            priority=ticket.priority,
            status=ticket.status,
            visibility=ticket.visibility,
            section=ticket.section,
            title=ticket.title,
            description=ticket.description,
            song_ref=ticket.song_ref,
            due_at=ticket.due_at,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            resolved_at=ticket.resolved_at,
            resolved_note=ticket.resolved_note,
            verified_at=ticket.verified_at,
            verified_by=ticket.verified_by,
            verified_note=ticket.verified_note,
        )


class CreateTicketResponse(BaseModel):
    """Response schema for ticket creation."""

    ticket: TicketResponse


class TicketsListResponse(BaseModel):
    """Paginated list of tickets.

    Follows pagination contract from systemprompt.md:
    - items: list of tickets
    - next_cursor: opaque string or null
    """

    items: list[TicketResponse]
    next_cursor: str | None


class UpdateTicketRequest(BaseModel):
    """Request schema for updating a ticket.

    Partial update - only provided fields are updated.
    Allowed by: owner OR leader/admin in scope.
    """

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    category: TicketCategory | None = None
    song_ref: str | None = Field(default=None, max_length=100)
    priority: Priority | None = None
    status: TicketStatus | None = None
    visibility: TicketVisibility | None = None
    section: str | None = Field(default=None, max_length=50)


class UpdateTicketResponse(BaseModel):
    """Response schema for ticket update."""

    ticket: TicketResponse


class ClaimTicketResponse(BaseModel):
    """Response schema for claiming a ticket."""

    ticket: TicketResponse


class TicketActivityResponse(BaseModel):
    """Ticket activity data in responses."""

    id: uuid.UUID
    ticket_id: uuid.UUID
    user_id: uuid.UUID
    type: TicketActivityType
    content: str | None
    old_status: TicketStatus | None
    new_status: TicketStatus | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(cls, activity: TicketActivity) -> TicketActivityResponse:
        """Create response from model."""
        return cls(
            id=activity.id,
            ticket_id=activity.ticket_id,
            user_id=activity.user_id,
            type=activity.type,
            content=activity.content,
            old_status=activity.old_status,
            new_status=activity.new_status,
            created_at=activity.created_at,
        )


class TicketActivitiesResponse(BaseModel):
    """Response for ticket activity timeline."""

    items: list[TicketActivityResponse]


# =============================================================================
# Transition and Verify Schemas (Milestone 8)
# =============================================================================


class TransitionTicketRequest(BaseModel):
    """Request schema for transitioning a ticket status.

    Transition rules:
    - Owner can: OPEN <-> IN_PROGRESS <-> BLOCKED
    - Owner can: IN_PROGRESS/BLOCKED -> RESOLVED (requires content)
    - VERIFIED only via /verify endpoint
    """

    to_status: TicketStatus
    content: str | None = Field(default=None, max_length=2000)


class TransitionTicketResponse(BaseModel):
    """Response schema for ticket transition."""

    ticket: TicketResponse


class VerifyTicketRequest(BaseModel):
    """Request schema for verifying a ticket.

    Only SECTION_LEADER (in scope) or ADMIN can verify.
    """

    content: str | None = Field(default=None, max_length=2000)


class VerifyTicketResponse(BaseModel):
    """Response schema for ticket verification."""

    ticket: TicketResponse

