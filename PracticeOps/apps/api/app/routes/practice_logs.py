"""Practice log routes.

POST /cycles/{cycle_id}/practice-logs - Create practice log
GET /cycles/{cycle_id}/practice-logs - List with me=true/false, pagination
PATCH /practice-logs/{id} - Update (owner only)

RBAC is enforced via dependencies and explicit checks.
Privacy: Members can only see their own logs unless they are leaders/admins.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import CurrentUser, require_non_demo
from app.core.errors import ForbiddenException, NotFoundException, ValidationException
from app.database import get_db
from app.models import (
    Assignment,
    AssignmentScope,
    PracticeLog,
    PracticeLogAssignment,
    Priority,
    RehearsalCycle,
    Role,
    TeamMembership,
    TicketCategory,
    TicketVisibility,
    User,
)
from app.schemas.practice_logs import (
    CreatePracticeLogRequest,
    CreatePracticeLogResponse,
    PracticeLogAssignmentResponse,
    PracticeLogResponse,
    PracticeLogsListResponse,
    SuggestedTicket,
    UpdatePracticeLogRequest,
    UpdatePracticeLogResponse,
)

# Router for cycle-scoped practice log endpoints
cycle_router = APIRouter(prefix="/cycles/{cycle_id}/practice-logs", tags=["practice-logs"])

# Router for practice log-specific endpoints
practice_log_router = APIRouter(prefix="/practice-logs", tags=["practice-logs"])


# =============================================================================
# Cursor encoding/decoding for pagination
# =============================================================================


class CursorData:
    """Cursor data for practice log pagination."""

    def __init__(self, occurred_at: datetime, id_: uuid.UUID) -> None:
        self.occurred_at = occurred_at
        self.id = id_


def encode_practice_log_cursor(occurred_at: datetime, id_: uuid.UUID) -> str:
    """Encode pagination cursor as base64 JSON.

    Cursor contains sort keys for deterministic pagination:
    - occurred_at (DESC)
    - id (tie-breaker)
    """
    data = {
        "occurred_at": occurred_at.isoformat(),
        "id": str(id_),
    }
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_practice_log_cursor(cursor: str) -> CursorData | None:
    """Decode pagination cursor from base64 JSON.

    Returns None if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return CursorData(
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            id_=uuid.UUID(data["id"]),
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# =============================================================================
# Helper functions
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


async def _validate_assignment_ids(
    assignment_ids: list[uuid.UUID],
    cycle_id: uuid.UUID,
    user_section: str | None,
    db: AsyncSession,
) -> list[Assignment]:
    """Validate that all assignment IDs exist, belong to the cycle, and are visible to user.

    Visibility rules (from Milestone 5):
    - TEAM scope assignments are always visible
    - SECTION scope assignments only visible if section matches user's section
    """
    if not assignment_ids:
        return []

    # Fetch all assignments
    result = await db.execute(
        select(Assignment).where(Assignment.id.in_(assignment_ids))
    )
    assignments = list(result.scalars().all())

    # Check all IDs were found
    found_ids = {a.id for a in assignments}
    missing_ids = set(assignment_ids) - found_ids
    if missing_ids:
        raise ValidationException(
            f"Assignment IDs not found: {list(missing_ids)}",
            field="assignment_ids",
        )

    # Validate each assignment
    for assignment in assignments:
        # Check cycle (implicitly validates team via cycle)
        if assignment.cycle_id != cycle_id:
            raise ValidationException(
                f"Assignment {assignment.id} belongs to a different cycle",
                field="assignment_ids",
            )

        # Check visibility: SECTION assignments only visible to user's section
        if assignment.scope == AssignmentScope.SECTION and assignment.section != user_section:
            raise ValidationException(
                f"Assignment {assignment.id} is not visible to your section",
                field="assignment_ids",
            )

    return assignments


def _build_practice_log_response(
    log: PracticeLog,
    assignment_details: list[PracticeLogAssignmentResponse] | None = None,
) -> PracticeLogResponse:
    """Build a practice log response with assignment details."""
    return PracticeLogResponse.from_model(log, assignment_details or [])


def _build_suggested_ticket(cycle_date: datetime | None) -> SuggestedTicket:
    """Build a suggested ticket object for blocked practice."""
    due_date = cycle_date.strftime("%Y-%m-%d") if cycle_date else datetime.now(UTC).strftime("%Y-%m-%d")
    return SuggestedTicket(
        title_suggestion="Practice blocker - needs attention",
        due_date=due_date,
        visibility_default=TicketVisibility.PRIVATE,
        priority_default=Priority.MEDIUM,
        category_default=TicketCategory.OTHER,
    )


# =============================================================================
# POST /cycles/{cycle_id}/practice-logs
# =============================================================================


@cycle_router.post("", response_model=CreatePracticeLogResponse, status_code=status.HTTP_201_CREATED)
async def create_practice_log(
    cycle_id: uuid.UUID,
    request: CreatePracticeLogRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> CreatePracticeLogResponse:
    """Create a practice log for a cycle.

    Rules:
    - occurred_at defaults to now
    - duration_min must be 1..600
    - rating_1_5 if provided must be 1..5
    - assignment_ids must exist, belong to same cycle, and be visible to user
    - suggested_ticket is included only if blocked_flag=true
    """
    # Get cycle to derive team_id
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Validate assignment IDs
    assignments = await _validate_assignment_ids(
        request.assignment_ids,
        cycle_id,
        membership.section,
        db,
    )

    # Set occurred_at to now if not provided
    occurred_at = request.occurred_at or datetime.now(UTC)

    # Create practice log
    practice_log = PracticeLog(
        user_id=current_user.id,
        team_id=cycle.team_id,
        cycle_id=cycle_id,
        duration_minutes=request.duration_min,
        rating_1_5=request.rating_1_5,
        blocked_flag=request.blocked_flag,
        notes=request.notes,
        occurred_at=occurred_at,
    )
    db.add(practice_log)
    await db.flush()  # Get log ID for join table

    # Create practice log assignments (join table rows)
    for assignment in assignments:
        pla = PracticeLogAssignment(
            practice_log_id=practice_log.id,
            assignment_id=assignment.id,
        )
        db.add(pla)

    await db.commit()
    await db.refresh(practice_log)

    # Build assignment details for response
    assignment_details = [
        PracticeLogAssignmentResponse(
            id=a.id,
            title=a.title,
            type=a.type.value,
        )
        for a in assignments
    ]

    # Build response
    response_log = _build_practice_log_response(practice_log, assignment_details)

    # Include suggested_ticket only if blocked_flag=true
    suggested_ticket = None
    if request.blocked_flag:
        suggested_ticket = _build_suggested_ticket(cycle.date)

    return CreatePracticeLogResponse(
        practice_log=response_log,
        suggested_ticket=suggested_ticket,
    )


# =============================================================================
# GET /cycles/{cycle_id}/practice-logs
# =============================================================================


@cycle_router.get("", response_model=PracticeLogsListResponse)
async def list_practice_logs(
    cycle_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    me: Annotated[bool, Query()] = True,
    section: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> PracticeLogsListResponse:
    """List practice logs for a cycle.

    Who can access:
    - me=true: any member → returns only their logs
    - me=false:
      - SECTION_LEADER → returns logs for their section only
      - ADMIN → returns logs for whole team

    Sorting (mandatory):
    1. occurred_at DESC
    2. id (tie-breaker)
    """
    # Get cycle to derive team_id
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Build base query
    query = (
        select(PracticeLog)
        .where(PracticeLog.cycle_id == cycle_id)
        .options(selectinload(PracticeLog.assignments).selectinload(PracticeLogAssignment.assignment))
    )

    # Apply access control
    if me:
        # Only show user's own logs
        query = query.where(PracticeLog.user_id == current_user.id)
    else:
        # me=false: Check role-based access
        if membership.role == Role.MEMBER:
            raise ForbiddenException("Members can only view their own practice logs (use me=true)")

        if membership.role == Role.SECTION_LEADER:
            # Section leader can only see logs from their section members
            if section and section != membership.section:
                raise ForbiddenException("Section leaders can only view logs from their own section")

            # Get all user IDs in the leader's section
            section_members_query = select(TeamMembership.user_id).where(
                TeamMembership.team_id == cycle.team_id,
                TeamMembership.section == membership.section,
            )
            section_member_result = await db.execute(section_members_query)
            section_member_ids = [row[0] for row in section_member_result.fetchall()]

            query = query.where(PracticeLog.user_id.in_(section_member_ids))

        elif membership.role == Role.ADMIN:
            # Admin can see all logs, optionally filtered by section
            if section:
                # Get user IDs in the specified section
                section_members_query = select(TeamMembership.user_id).where(
                    TeamMembership.team_id == cycle.team_id,
                    TeamMembership.section == section,
                )
                section_member_result = await db.execute(section_members_query)
                section_member_ids = [row[0] for row in section_member_result.fetchall()]

                query = query.where(PracticeLog.user_id.in_(section_member_ids))
            # If no section filter, admin sees all

    # Apply cursor filter if provided
    if cursor:
        decoded = decode_practice_log_cursor(cursor)
        if decoded:
            # Sorting: occurred_at DESC, id DESC (for tie-breaker)
            # "After cursor" means: occurred_at < cursor OR (occurred_at == cursor AND id < cursor_id)
            query = query.where(
                or_(
                    PracticeLog.occurred_at < decoded.occurred_at,
                    (PracticeLog.occurred_at == decoded.occurred_at)
                    & (PracticeLog.id < decoded.id),
                )
            )

    # Apply ordering: occurred_at DESC, id DESC (tie-breaker)
    query = query.order_by(PracticeLog.occurred_at.desc(), PracticeLog.id.desc())

    # Fetch limit + 1 to determine if there's a next page
    query = query.limit(limit + 1)
    result = await db.execute(query)
    logs = list(result.scalars().unique().all())

    # Check if there's a next page
    has_next = len(logs) > limit
    if has_next:
        logs = logs[:limit]

    # Build response items with assignment details
    items: list[PracticeLogResponse] = []
    for log in logs:
        assignment_details = [
            PracticeLogAssignmentResponse(
                id=pla.assignment.id,
                title=pla.assignment.title,
                type=pla.assignment.type.value,
            )
            for pla in log.assignments
            if pla.assignment is not None
        ]
        items.append(_build_practice_log_response(log, assignment_details))

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and logs:
        last = logs[-1]
        next_cursor = encode_practice_log_cursor(last.occurred_at, last.id)

    return PracticeLogsListResponse(items=items, next_cursor=next_cursor)


# =============================================================================
# PATCH /practice-logs/{id}
# =============================================================================


@practice_log_router.patch("/{practice_log_id}", response_model=UpdatePracticeLogResponse)
async def update_practice_log(
    practice_log_id: uuid.UUID,
    request: UpdatePracticeLogRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> UpdatePracticeLogResponse:
    """Update a practice log.

    Owner only - only the user who created the log can update it.
    If assignment_ids updated, re-validate visibility + correct cycle/team.
    """
    # Get practice log with assignments
    result = await db.execute(
        select(PracticeLog)
        .where(PracticeLog.id == practice_log_id)
        .options(selectinload(PracticeLog.assignments).selectinload(PracticeLogAssignment.assignment))
    )
    practice_log = result.scalar_one_or_none()

    if practice_log is None:
        raise NotFoundException("Practice log not found")

    # Owner only check
    if practice_log.user_id != current_user.id:
        raise ForbiddenException("Only the owner can update this practice log")

    # Get membership for assignment validation
    membership = await _require_membership(practice_log.team_id, current_user.id, db)

    # Apply partial updates
    if request.occurred_at is not None:
        practice_log.occurred_at = request.occurred_at

    if request.duration_min is not None:
        practice_log.duration_minutes = request.duration_min

    if request.notes is not None:
        practice_log.notes = request.notes

    if request.rating_1_5 is not None:
        practice_log.rating_1_5 = request.rating_1_5

    if request.blocked_flag is not None:
        practice_log.blocked_flag = request.blocked_flag

    # Handle assignment_ids update
    new_assignment_details: list[PracticeLogAssignmentResponse] | None = None
    if request.assignment_ids is not None:
        # Validate new assignment IDs
        if practice_log.cycle_id is None:
            raise ValidationException(
                "Cannot update assignments for a log without a cycle",
                field="assignment_ids",
            )

        assignments = await _validate_assignment_ids(
            request.assignment_ids,
            practice_log.cycle_id,
            membership.section,
            db,
        )

        # Remove existing assignments
        for pla in practice_log.assignments:
            await db.delete(pla)

        # Add new assignments
        for assignment in assignments:
            pla = PracticeLogAssignment(
                practice_log_id=practice_log.id,
                assignment_id=assignment.id,
            )
            db.add(pla)

        new_assignment_details = [
            PracticeLogAssignmentResponse(
                id=a.id,
                title=a.title,
                type=a.type.value,
            )
            for a in assignments
        ]

    await db.commit()

    # Re-fetch if assignment_ids was not updated (to get current assignments)
    if new_assignment_details is None:
        await db.refresh(practice_log, attribute_names=["assignments"])
        new_assignment_details = [
            PracticeLogAssignmentResponse(
                id=pla.assignment.id,
                title=pla.assignment.title,
                type=pla.assignment.type.value,
            )
            for pla in practice_log.assignments
            if pla.assignment is not None
        ]

    return UpdatePracticeLogResponse(
        practice_log=_build_practice_log_response(practice_log, new_assignment_details)
    )

