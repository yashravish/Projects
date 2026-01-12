"""Assignment routes.

POST /cycles/{cycle_id}/assignments - Create assignment (ADMIN, SECTION_LEADER with constraints)
GET /cycles/{cycle_id}/assignments - List assignments with visibility rules and pagination
PATCH /assignments/{id} - Update assignment (creator or ADMIN)
DELETE /assignments/{id} - Delete assignment (ADMIN only)

RBAC is enforced via dependencies, not inline logic.
Visibility rules are enforced in queries.
"""

import base64
import json
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ForbiddenException, NotFoundException, ValidationException
from app.database import get_db
from app.models import Assignment, AssignmentScope, Priority, RehearsalCycle, Role, TeamMembership
from app.schemas.assignments import (
    AssignmentResponse,
    AssignmentsListResponse,
    CreateAssignmentRequest,
    CreateAssignmentResponse,
    UpdateAssignmentRequest,
    UpdateAssignmentResponse,
)


class CursorData(TypedDict):
    """Type for decoded cursor data."""

    priority_order: int
    due_at: datetime | None
    created_at: datetime
    id: uuid.UUID

# Router for cycle-scoped assignment endpoints
cycle_router = APIRouter(prefix="/cycles/{cycle_id}/assignments", tags=["assignments"])

# Router for assignment-specific endpoints
assignment_router = APIRouter(prefix="/assignments", tags=["assignments"])


# =============================================================================
# Priority ordering for sorting
# =============================================================================

PRIORITY_ORDER = {
    Priority.BLOCKING: 0,  # Highest priority (first in DESC)
    Priority.MEDIUM: 1,
    Priority.LOW: 2,  # Lowest priority (last in DESC)
}


def encode_assignment_cursor(
    priority: Priority, due_at: datetime | None, created_at: datetime, id_: uuid.UUID
) -> str:
    """Encode pagination cursor as base64 JSON for assignments.

    Cursor contains all sort keys for deterministic pagination:
    - priority (as numeric order)
    - due_at
    - created_at
    - id (tie-breaker)
    """
    data = {
        "priority_order": PRIORITY_ORDER[priority],
        "due_at": due_at.isoformat() if due_at else None,
        "created_at": created_at.isoformat(),
        "id": str(id_),
    }
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_assignment_cursor(cursor: str) -> CursorData | None:
    """Decode pagination cursor from base64 JSON.

    Returns None if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return CursorData(
            priority_order=data["priority_order"],
            due_at=datetime.fromisoformat(data["due_at"]) if data["due_at"] else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            id=uuid.UUID(data["id"]),
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# =============================================================================
# Helper functions for RBAC and membership
# =============================================================================


async def _get_membership(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership | None:
    """Get user's membership for a team."""
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _require_membership(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership:
    """Require user to be a member of the team."""
    membership = await _get_membership(team_id, user_id, db)
    if membership is None:
        raise ForbiddenException("Not a member of this team")
    return membership


async def _get_cycle_with_team(
    cycle_id: uuid.UUID,
    db: AsyncSession,
) -> RehearsalCycle | None:
    """Get cycle with team_id for RBAC checks."""
    result = await db.execute(
        select(RehearsalCycle).where(RehearsalCycle.id == cycle_id)
    )
    return result.scalar_one_or_none()


async def _get_assignment_with_cycle(
    assignment_id: uuid.UUID,
    db: AsyncSession,
) -> tuple[Assignment, RehearsalCycle] | None:
    """Get assignment with its cycle for RBAC checks."""
    result = await db.execute(
        select(Assignment, RehearsalCycle)
        .join(RehearsalCycle, Assignment.cycle_id == RehearsalCycle.id)
        .where(Assignment.id == assignment_id)
    )
    row = result.first()
    if row is None:
        return None
    return row[0], row[1]


# =============================================================================
# POST /cycles/{cycle_id}/assignments
# =============================================================================


@cycle_router.post("", response_model=CreateAssignmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assignment(
    cycle_id: uuid.UUID,
    request: CreateAssignmentRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateAssignmentResponse:
    """Create a new assignment for a cycle.

    Who can create:
    - ADMIN: can create TEAM or SECTION
    - SECTION_LEADER: can create SECTION only for their own section

    Server sets (mandatory):
    - team_id from the referenced cycle
    - created_by from auth user
    - due_date = cycle.date

    Validation:
    - scope=TEAM must not require section
    - scope=SECTION must include a valid section
    - SECTION_LEADER creating for a different section must be rejected
    """
    # Get cycle to derive team_id and due_date
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership in the team
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # RBAC: Check if user can create assignments
    if membership.role == Role.MEMBER:
        raise ForbiddenException("Members cannot create assignments")

    if membership.role == Role.SECTION_LEADER:
        # Section leaders can only create SECTION assignments for their own section
        if request.scope == AssignmentScope.TEAM:
            raise ForbiddenException("Section leaders cannot create TEAM assignments")
        if request.section != membership.section:
            raise ForbiddenException(
                f"Section leaders can only create assignments for their own section ({membership.section})"
            )

    # ADMIN can create any scope/section combination (already validated by Pydantic)

    # Create assignment with server-set fields
    assignment = Assignment(
        cycle_id=cycle_id,
        created_by=current_user.id,
        type=request.type,
        scope=request.scope,
        priority=request.priority,
        section=request.section,
        title=request.title,
        song_ref=request.song_ref,
        description=request.notes,  # Map notes to description
        due_at=cycle.date,  # Server sets due_date from cycle.date
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)

    return CreateAssignmentResponse(
        assignment=AssignmentResponse.from_model(assignment)
    )


# =============================================================================
# GET /cycles/{cycle_id}/assignments
# =============================================================================


@cycle_router.get("", response_model=AssignmentsListResponse)
async def list_assignments(
    cycle_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    scope: Annotated[AssignmentScope | None, Query()] = None,
    section: Annotated[str | None, Query()] = None,
    priority: Annotated[Priority | None, Query()] = None,
    type: Annotated[str | None, Query(alias="type")] = None,
    song_ref: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> AssignmentsListResponse:
    """List assignments for a cycle with visibility rules.

    Visibility rules (mandatory):
    - Members see:
      - TEAM assignments
      - SECTION assignments only for their section
    - Leaders/Admins follow the same visibility rules

    Sorting (mandatory):
    1. priority DESC (BLOCKING first)
    2. due_at ASC
    3. created_at DESC
    4. id (tie-breaker)
    """
    # Get cycle to derive team_id
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (also validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Build base query with visibility rules
    query = select(Assignment).where(Assignment.cycle_id == cycle_id)

    # Apply visibility filter:
    # - TEAM scope assignments are always visible
    # - SECTION scope assignments only visible if section matches user's section
    visibility_filter = or_(
        Assignment.scope == AssignmentScope.TEAM,
        (Assignment.scope == AssignmentScope.SECTION) & (Assignment.section == membership.section),
    )
    query = query.where(visibility_filter)

    # Apply optional filters
    if scope is not None:
        query = query.where(Assignment.scope == scope)
    if section is not None:
        query = query.where(Assignment.section == section)
    if priority is not None:
        query = query.where(Assignment.priority == priority)
    if type is not None:
        query = query.where(Assignment.type == type)
    if song_ref is not None:
        query = query.where(Assignment.song_ref == song_ref)

    # Apply ordering: priority DESC, due_at ASC, created_at DESC, id ASC
    # Note: PostgreSQL ENUM ordering uses the order defined in the type
    # Priority enum: LOW, MEDIUM, BLOCKING - we need BLOCKING first (highest)
    # We'll use CASE expression for proper priority ordering
    from sqlalchemy import case

    priority_order = case(
        (Assignment.priority == Priority.BLOCKING, 0),
        (Assignment.priority == Priority.MEDIUM, 1),
        (Assignment.priority == Priority.LOW, 2),
    )

    query = query.order_by(
        priority_order.asc(),  # BLOCKING first (0)
        Assignment.due_at.asc().nulls_last(),  # Earliest due first
        Assignment.created_at.desc(),  # Most recent first
        Assignment.id.asc(),  # Deterministic tie-breaker
    )

    # Fetch more rows than limit to handle cursor filtering
    fetch_limit = limit + 1
    if cursor:
        fetch_limit = (limit + 1) * 2  # Fetch extra to handle cursor filtering

    query = query.limit(fetch_limit)
    result = await db.execute(query)
    all_assignments = list(result.scalars().all())

    # If cursor provided, filter to rows after cursor position
    if cursor:
        decoded = decode_assignment_cursor(cursor)
        if decoded:
            cursor_priority_order = decoded["priority_order"]
            cursor_due_at = decoded["due_at"]
            cursor_created_at = decoded["created_at"]
            cursor_id = decoded["id"]

            filtered_assignments: list[Assignment] = []
            for a in all_assignments:
                a_priority_order = PRIORITY_ORDER[a.priority]

                # Compare by sort keys in order
                if a_priority_order > cursor_priority_order:
                    filtered_assignments.append(a)
                elif a_priority_order == cursor_priority_order:
                    # Compare due_at ASC (None is treated as "infinite" - last)
                    a_due = a.due_at
                    c_due = cursor_due_at

                    if a_due is None and c_due is None:
                        due_compare = 0
                    elif a_due is None:
                        due_compare = 1  # a is "after" c
                    elif c_due is None:
                        due_compare = -1
                    else:
                        due_compare = (a_due > c_due) - (a_due < c_due)

                    if due_compare > 0:
                        filtered_assignments.append(a)
                    elif due_compare == 0:
                        # Compare created_at DESC
                        if a.created_at < cursor_created_at:
                            filtered_assignments.append(a)
                        elif a.created_at == cursor_created_at and a.id > cursor_id:
                            # Compare id ASC (tie-breaker)
                            filtered_assignments.append(a)

            all_assignments = filtered_assignments

    # Check if there's a next page
    has_next = len(all_assignments) > limit
    if has_next:
        all_assignments = all_assignments[:limit]

    # Build response items
    items = [AssignmentResponse.from_model(a) for a in all_assignments]

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and all_assignments:
        last = all_assignments[-1]
        next_cursor = encode_assignment_cursor(
            last.priority, last.due_at, last.created_at, last.id
        )

    return AssignmentsListResponse(items=items, next_cursor=next_cursor)


# =============================================================================
# PATCH /assignments/{id}
# =============================================================================


@assignment_router.patch("/{assignment_id}", response_model=UpdateAssignmentResponse)
async def update_assignment(
    assignment_id: uuid.UUID,
    request: UpdateAssignmentRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UpdateAssignmentResponse:
    """Update an assignment.

    Allowed: Creator OR ADMIN

    Partial update - only provided fields are updated.
    """
    # Get assignment with its cycle for team context
    assignment_data = await _get_assignment_with_cycle(assignment_id, db)
    if assignment_data is None:
        raise NotFoundException("Assignment not found")

    assignment, cycle = assignment_data

    # Get user's membership (also validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # RBAC: Creator or ADMIN can edit
    is_creator = assignment.created_by == current_user.id
    is_admin = membership.role == Role.ADMIN

    if not is_creator and not is_admin:
        raise ForbiddenException("Only the creator or an admin can edit this assignment")

    # Apply partial updates
    if request.title is not None:
        assignment.title = request.title
    if request.type is not None:
        assignment.type = request.type
    if request.priority is not None:
        assignment.priority = request.priority
    if request.song_ref is not None:
        assignment.song_ref = request.song_ref
    if request.notes is not None:
        assignment.description = request.notes
    if request.section is not None:
        # Validate section change doesn't break scope invariant
        if assignment.scope == AssignmentScope.TEAM:
            raise ValidationException(
                "Cannot set section on TEAM scope assignment",
                field="section",
            )
        assignment.section = request.section

    await db.commit()
    await db.refresh(assignment)

    return UpdateAssignmentResponse(
        assignment=AssignmentResponse.from_model(assignment)
    )


# =============================================================================
# DELETE /assignments/{id}
# =============================================================================


@assignment_router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assignment(
    assignment_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an assignment.

    Allowed: ADMIN only
    """
    # Get assignment with its cycle for team context
    assignment_data = await _get_assignment_with_cycle(assignment_id, db)
    if assignment_data is None:
        raise NotFoundException("Assignment not found")

    assignment, cycle = assignment_data

    # Get user's membership (also validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # RBAC: ADMIN only
    if membership.role != Role.ADMIN:
        raise ForbiddenException("Only admins can delete assignments")

    await db.delete(assignment)
    await db.commit()

