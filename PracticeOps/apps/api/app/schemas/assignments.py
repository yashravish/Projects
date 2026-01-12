"""Assignment schemas for request/response models.

Implements contracts for:
- POST /cycles/{cycle_id}/assignments
- GET /cycles/{cycle_id}/assignments
- PATCH /assignments/{id}
- DELETE /assignments/{id}
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from app.models.enums import AssignmentScope, AssignmentType, Priority

if TYPE_CHECKING:
    from app.models import Assignment


class CreateAssignmentRequest(BaseModel):
    """Request schema for creating an assignment.

    Server sets (mandatory):
    - team_id from the referenced cycle
    - created_by from auth user
    - due_date = cycle.date

    Validation rules:
    - scope=TEAM must not have section
    - scope=SECTION must include a valid section
    """

    title: str = Field(min_length=1, max_length=200)
    type: AssignmentType
    song_ref: str | None = Field(default=None, max_length=100)
    scope: AssignmentScope
    section: str | None = Field(default=None, max_length=50)
    priority: Priority
    notes: str | None = Field(default=None, description="Maps to description in model")

    @model_validator(mode="after")
    def validate_scope_section(self) -> CreateAssignmentRequest:
        """Validate scope and section combination."""
        if self.scope == AssignmentScope.TEAM and self.section is not None:
            raise ValueError("TEAM scope assignments must not have a section")
        if self.scope == AssignmentScope.SECTION and self.section is None:
            raise ValueError("SECTION scope assignments must include a section")
        return self


class AssignmentResponse(BaseModel):
    """Assignment data in responses."""

    id: uuid.UUID
    cycle_id: uuid.UUID
    created_by: uuid.UUID | None
    type: AssignmentType
    scope: AssignmentScope
    priority: Priority
    section: str | None
    title: str
    song_ref: str | None
    notes: str | None = Field(description="Maps from description in model")
    due_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_model(cls, assignment: Assignment) -> AssignmentResponse:
        """Create response from model, mapping description to notes."""
        return cls(
            id=assignment.id,
            cycle_id=assignment.cycle_id,
            created_by=assignment.created_by,
            type=assignment.type,
            scope=assignment.scope,
            priority=assignment.priority,
            section=assignment.section,
            title=assignment.title,
            song_ref=assignment.song_ref,
            notes=assignment.description,
            due_at=assignment.due_at,
            created_at=assignment.created_at,
            updated_at=assignment.updated_at,
        )


class CreateAssignmentResponse(BaseModel):
    """Response schema for assignment creation."""

    assignment: AssignmentResponse


class AssignmentsListResponse(BaseModel):
    """Paginated list of assignments.

    Follows pagination contract from systemprompt.md:
    - items: list of assignments
    - next_cursor: opaque string or null
    """

    items: list[AssignmentResponse]
    next_cursor: str | None


class UpdateAssignmentRequest(BaseModel):
    """Request schema for updating an assignment.

    Partial update - only provided fields are updated.
    Cannot change scope from TEAM to SECTION or vice versa without
    proper section handling.
    """

    title: str | None = Field(default=None, min_length=1, max_length=200)
    type: AssignmentType | None = None
    song_ref: str | None = Field(default=None, max_length=100)
    priority: Priority | None = None
    section: str | None = Field(default=None, max_length=50)
    notes: str | None = None


class UpdateAssignmentResponse(BaseModel):
    """Response schema for assignment update."""

    assignment: AssignmentResponse

