"""Ticket routes.

POST /cycles/{cycle_id}/tickets - Create ticket
GET /cycles/{cycle_id}/tickets - List tickets with visibility enforcement
POST /tickets/{id}/claim - Claim a claimable ticket (atomic)
PATCH /tickets/{id} - Update ticket (owner or leader in scope)
GET /tickets/{id}/activity - View ticket activity

RBAC and visibility are enforced via dependencies and shared helpers.
Visibility rules from systemprompt.md are strictly enforced.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from typing import Annotated, Any, TypedDict

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import case, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import CurrentUser, require_non_demo
from app.core.errors import (
    ConflictException,
    ForbiddenException,
    NotFoundException,
    ValidationException,
)
from app.database import get_db
from app.models import (
    Priority,
    RehearsalCycle,
    Role,
    TeamMembership,
    Ticket,
    TicketActivity,
    TicketActivityType,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
    User,
)
from app.schemas.tickets import (
    ClaimTicketResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    TicketActivitiesResponse,
    TicketActivityResponse,
    TicketResponse,
    TicketsListResponse,
    TransitionTicketRequest,
    TransitionTicketResponse,
    UpdateTicketRequest,
    UpdateTicketResponse,
    VerifyTicketRequest,
    VerifyTicketResponse,
)

# Router for cycle-scoped ticket endpoints
cycle_router = APIRouter(prefix="/cycles/{cycle_id}/tickets", tags=["tickets"])

# Router for ticket-specific endpoints
ticket_router = APIRouter(prefix="/tickets", tags=["tickets"])


# =============================================================================
# Priority ordering for sorting
# =============================================================================

PRIORITY_ORDER = {
    Priority.BLOCKING: 0,  # Highest priority (first in DESC)
    Priority.MEDIUM: 1,
    Priority.LOW: 2,  # Lowest priority (last in DESC)
}


class CursorData(TypedDict):
    """Type for decoded cursor data."""

    priority_order: int
    due_at: str | None
    updated_at: str
    id: str


def encode_ticket_cursor(
    priority: Priority, due_at: datetime | None, updated_at: datetime, id_: uuid.UUID
) -> str:
    """Encode pagination cursor as base64 JSON for tickets.

    Cursor contains all sort keys for deterministic pagination:
    - priority (as numeric order)
    - due_at ASC
    - updated_at DESC
    - id (tie-breaker)
    """
    data: CursorData = {
        "priority_order": PRIORITY_ORDER[priority],
        "due_at": due_at.isoformat() if due_at else None,
        "updated_at": updated_at.isoformat(),
        "id": str(id_),
    }
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_ticket_cursor(cursor: str) -> CursorData | None:
    """Decode pagination cursor from base64 JSON.

    Returns None if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return CursorData(
            priority_order=data["priority_order"],
            due_at=data["due_at"],
            updated_at=data["updated_at"],
            id=data["id"],
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# =============================================================================
# Helper functions for RBAC and visibility
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


def _can_view_ticket(
    ticket: Ticket,
    membership: TeamMembership,
    user_id: uuid.UUID,
) -> bool:
    """Check if user can view a ticket based on visibility rules.

    Visibility rules from systemprompt.md:
    - PRIVATE: Only owner + Admin + SectionLeader (in owner's section)
    - SECTION: Members in that section + its SectionLeader + Admin
    - TEAM: All team members + leaders
    """
    if membership.role == Role.ADMIN:
        return True

    if ticket.visibility == TicketVisibility.TEAM:
        # All team members can view
        return True

    if ticket.visibility == TicketVisibility.SECTION:
        # Members in that section (+ admin already covered)
        # Section leaders can also view their section's tickets
        return membership.section == ticket.section

    if ticket.visibility == TicketVisibility.PRIVATE:
        # Only owner can view
        if ticket.owner_id == user_id:
            return True
        # Creator can view their own tickets
        if ticket.created_by == user_id:
            return True
        # Section leader can view if they lead the owner's section
        return (
            membership.role == Role.SECTION_LEADER
            and ticket.section is not None
            and membership.section == ticket.section
        )

    return False


def _can_edit_ticket(
    ticket: Ticket,
    membership: TeamMembership,
    user_id: uuid.UUID,
) -> bool:
    """Check if user can edit a ticket.

    Allowed:
    - Owner
    - Admin
    - Section leader if ticket is in their section
    """
    if membership.role == Role.ADMIN:
        return True

    if ticket.owner_id == user_id:
        return True

    if ticket.created_by == user_id:
        return True

    return (
        membership.role == Role.SECTION_LEADER
        and ticket.section is not None
        and membership.section == ticket.section
    )


def _build_visibility_filter(
    membership: TeamMembership,
    user_id: uuid.UUID,
) -> bool | Any:
    """Build SQLAlchemy filter for ticket visibility.

    Returns a filter expression that limits tickets to only those the user can see.
    """
    if membership.role == Role.ADMIN:
        # Admin can see all
        return True

    filters = []

    # TEAM visibility - all can see
    filters.append(Ticket.visibility == TicketVisibility.TEAM)

    # SECTION visibility - only if member's section matches
    if membership.section:
        filters.append(
            (Ticket.visibility == TicketVisibility.SECTION)
            & (Ticket.section == membership.section)
        )

    # PRIVATE visibility - only owner/creator
    filters.append(
        (Ticket.visibility == TicketVisibility.PRIVATE)
        & ((Ticket.owner_id == user_id) | (Ticket.created_by == user_id))
    )

    # Section leader can also see PRIVATE tickets in their section
    if membership.role == Role.SECTION_LEADER and membership.section:
        filters.append(
            (Ticket.visibility == TicketVisibility.PRIVATE)
            & (Ticket.section == membership.section)
        )

    return or_(*filters)


# =============================================================================
# POST /cycles/{cycle_id}/tickets
# =============================================================================


@cycle_router.post("", response_model=CreateTicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    cycle_id: uuid.UUID,
    request: CreateTicketRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> CreateTicketResponse:
    """Create a new ticket for a cycle.

    Rules:
    - Members can create tickets with visibility: PRIVATE, SECTION, TEAM
    - If visibility=SECTION, section must be provided
    - Leaders (ADMIN/SECTION_LEADER) can create claimable tickets with owner_id=null
    - Server sets: due_date, status=OPEN, created_by
    """
    # Get cycle to derive team_id and due_date
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Validate claimable permissions - only leaders can create claimable tickets
    if request.claimable and membership.role not in [Role.ADMIN, Role.SECTION_LEADER]:
        raise ForbiddenException("Only leaders can create claimable tickets")

    # Determine owner_id
    owner_id: uuid.UUID | None
    if request.claimable:
        owner_id = None  # Claimable tickets have no owner until claimed
    elif request.owner_id:
        owner_id = request.owner_id
    else:
        owner_id = current_user.id  # Default to creator

    # Set section for non-team visibility if not provided
    section = request.section
    if request.visibility == TicketVisibility.SECTION and section is None:
        # This should be caught by Pydantic validation, but double-check
        raise ValidationException(
            "SECTION visibility requires a section to be specified",
            field="section",
        )

    # If visibility is PRIVATE or SECTION and no section provided, use member's section
    if request.visibility != TicketVisibility.TEAM and section is None:
        section = membership.section

    # Create ticket with explicit ID
    ticket_id = uuid.uuid4()
    ticket = Ticket(
        id=ticket_id,
        team_id=cycle.team_id,
        cycle_id=cycle_id,
        owner_id=owner_id,
        created_by=current_user.id,
        claimable=request.claimable,
        category=request.category,
        priority=request.priority,
        status=TicketStatus.OPEN,  # Server sets
        visibility=request.visibility,
        section=section,
        title=request.title,
        description=request.description,
        song_ref=request.song_ref,
        due_at=cycle.date,  # Server sets from cycle
    )
    db.add(ticket)

    # Create CREATED activity (using explicit ticket_id)
    activity = TicketActivity(
        ticket_id=ticket_id,
        user_id=current_user.id,
        type=TicketActivityType.CREATED,
    )
    db.add(activity)

    await db.commit()
    await db.refresh(ticket)

    return CreateTicketResponse(ticket=TicketResponse.from_model(ticket))


# =============================================================================
# GET /cycles/{cycle_id}/tickets
# =============================================================================


@cycle_router.get("", response_model=TicketsListResponse)
async def list_tickets(
    cycle_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    status_filter: Annotated[TicketStatus | None, Query(alias="status")] = None,
    priority: Annotated[Priority | None, Query()] = None,
    category: Annotated[TicketCategory | None, Query()] = None,
    visibility: Annotated[TicketVisibility | None, Query()] = None,
    section: Annotated[str | None, Query()] = None,
    song_ref: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> TicketsListResponse:
    """List tickets for a cycle with visibility enforcement.

    Sorting (mandatory):
    1. priority DESC (BLOCKING first)
    2. due_at ASC
    3. updated_at DESC
    4. id (tie-breaker)
    """
    # Get cycle to derive team_id
    cycle = await _get_cycle_with_team(cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Build base query with visibility filter
    query = select(Ticket).where(Ticket.cycle_id == cycle_id)

    # Apply visibility filter
    visibility_filter = _build_visibility_filter(membership, current_user.id)
    if visibility_filter is not True:
        query = query.where(visibility_filter)  # type: ignore[arg-type]

    # Apply optional filters
    if status_filter is not None:
        query = query.where(Ticket.status == status_filter)
    if priority is not None:
        query = query.where(Ticket.priority == priority)
    if category is not None:
        query = query.where(Ticket.category == category)
    if visibility is not None:
        query = query.where(Ticket.visibility == visibility)
    if section is not None:
        query = query.where(Ticket.section == section)
    if song_ref is not None:
        query = query.where(Ticket.song_ref == song_ref)

    # Build priority ordering case expression
    priority_order = case(
        (Ticket.priority == Priority.BLOCKING, 0),
        (Ticket.priority == Priority.MEDIUM, 1),
        (Ticket.priority == Priority.LOW, 2),
    )

    # Apply ordering: priority DESC, due_at ASC, updated_at DESC, id ASC
    query = query.order_by(
        priority_order.asc(),  # BLOCKING first (0)
        Ticket.due_at.asc().nulls_last(),
        Ticket.updated_at.desc(),
        Ticket.id.asc(),  # Deterministic tie-breaker
    )

    # Apply cursor filter if provided
    if cursor:
        decoded = decode_ticket_cursor(cursor)
        if decoded:
            cursor_priority_order = decoded["priority_order"]
            cursor_due_at = datetime.fromisoformat(decoded["due_at"]) if decoded["due_at"] else None
            cursor_updated_at = datetime.fromisoformat(decoded["updated_at"])
            cursor_id = uuid.UUID(decoded["id"])

            # Complex cursor comparison for multi-column sort
            # We fetch extra and filter in Python for correctness
            pass  # Handled below

    # Fetch limit + extra for cursor filtering
    fetch_limit = limit + 1
    if cursor:
        fetch_limit = (limit + 1) * 2

    query = query.limit(fetch_limit)
    result = await db.execute(query)
    all_tickets = list(result.scalars().all())

    # Apply cursor filter in Python
    if cursor:
        decoded = decode_ticket_cursor(cursor)
        if decoded:
            cursor_priority_order = decoded["priority_order"]
            cursor_due_at = datetime.fromisoformat(decoded["due_at"]) if decoded["due_at"] else None
            cursor_updated_at = datetime.fromisoformat(decoded["updated_at"])
            cursor_id = uuid.UUID(decoded["id"])

            filtered_tickets = []
            for t in all_tickets:
                t_priority_order = PRIORITY_ORDER[t.priority]

                # Compare by sort keys: priority ASC, due_at ASC, updated_at DESC, id ASC
                if t_priority_order > cursor_priority_order:
                    filtered_tickets.append(t)
                elif t_priority_order == cursor_priority_order:
                    # Compare due_at ASC
                    t_due = t.due_at
                    c_due = cursor_due_at

                    if t_due is None and c_due is None:
                        due_compare = 0
                    elif t_due is None:
                        due_compare = 1
                    elif c_due is None:
                        due_compare = -1
                    else:
                        due_compare = (t_due > c_due) - (t_due < c_due)

                    if due_compare > 0 or (
                        due_compare == 0
                        and (
                            # Compare updated_at DESC
                            t.updated_at < cursor_updated_at
                            or (t.updated_at == cursor_updated_at and t.id > cursor_id)
                        )
                    ):
                        filtered_tickets.append(t)

            all_tickets = filtered_tickets

    # Check if there's a next page
    has_next = len(all_tickets) > limit
    if has_next:
        all_tickets = all_tickets[:limit]

    # Build response items
    items = [TicketResponse.from_model(t) for t in all_tickets]

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and all_tickets:
        last = all_tickets[-1]
        next_cursor = encode_ticket_cursor(last.priority, last.due_at, last.updated_at, last.id)

    return TicketsListResponse(items=items, next_cursor=next_cursor)


# =============================================================================
# POST /tickets/{id}/claim
# =============================================================================


@ticket_router.post("/{ticket_id}/claim", response_model=ClaimTicketResponse)
async def claim_ticket(
    ticket_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> ClaimTicketResponse:
    """Claim a claimable ticket.

    Rules:
    - Only allowed if claimable=true and owner_id IS NULL
    - User must be allowed to view/act on the ticket
    - On success: set owner_id, claimed_by, create CLAIMED activity
    - Double-claim returns CONFLICT
    - Uses atomic update for concurrency safety
    """
    # Fetch ticket
    result = await db.execute(
        select(Ticket).where(Ticket.id == ticket_id).with_for_update()
    )
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise NotFoundException("Ticket not found")

    # Get cycle to find team
    if ticket.cycle_id is None:
        raise NotFoundException("Ticket has no associated cycle")

    cycle = await _get_cycle_with_team(ticket.cycle_id, db)
    if cycle is None:
        raise NotFoundException("Cycle not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(cycle.team_id, current_user.id, db)

    # Check visibility - user must be able to see the ticket
    if not _can_view_ticket(ticket, membership, current_user.id):
        raise ForbiddenException("You do not have access to this ticket")

    # Check if ticket is claimable
    if not ticket.claimable:
        raise ForbiddenException("This ticket is not claimable")

    # Check if already claimed (owner_id set)
    if ticket.owner_id is not None:
        raise ConflictException("Ticket has already been claimed")

    # Atomic claim using conditional update
    stmt = (
        update(Ticket)
        .where(Ticket.id == ticket_id)
        .where(Ticket.owner_id.is_(None))  # Ensure still unclaimed
        .values(
            owner_id=current_user.id,
            claimed_by=current_user.id,
        )
        .returning(Ticket.id)
    )
    claim_result = await db.execute(stmt)
    claimed_id = claim_result.scalar_one_or_none()

    if claimed_id is None:
        # Race condition - someone else claimed it
        raise ConflictException("Ticket has already been claimed")

    # Create CLAIMED activity
    activity = TicketActivity(
        ticket_id=ticket_id,
        user_id=current_user.id,
        type=TicketActivityType.CLAIMED,
    )
    db.add(activity)

    await db.commit()

    # Refresh ticket to get updated values
    await db.refresh(ticket)

    return ClaimTicketResponse(ticket=TicketResponse.from_model(ticket))


# =============================================================================
# PATCH /tickets/{id}
# =============================================================================


@ticket_router.patch("/{ticket_id}", response_model=UpdateTicketResponse)
async def update_ticket(
    ticket_id: uuid.UUID,
    request: UpdateTicketRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> UpdateTicketResponse:
    """Update a ticket.

    Allowed: Owner OR Leader/Admin in scope
    Respects invariants (SECTION visibility requires section).
    """
    # Fetch ticket
    result = await db.execute(
        select(Ticket).where(Ticket.id == ticket_id)
    )
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise NotFoundException("Ticket not found")

    # Get team_id from ticket
    team_id = ticket.team_id

    # Get user's membership (validates team access)
    membership = await _require_membership(team_id, current_user.id, db)

    # Check edit permissions
    if not _can_edit_ticket(ticket, membership, current_user.id):
        raise ForbiddenException("You do not have permission to edit this ticket")

    # Apply partial updates
    if request.title is not None:
        ticket.title = request.title
    if request.description is not None:
        ticket.description = request.description
    if request.category is not None:
        ticket.category = request.category
    if request.song_ref is not None:
        ticket.song_ref = request.song_ref
    if request.priority is not None:
        ticket.priority = request.priority
    if request.status is not None:
        ticket.status = request.status
    if request.visibility is not None:
        # Validate SECTION visibility requires section
        if request.visibility == TicketVisibility.SECTION:
            section_to_use = request.section if request.section is not None else ticket.section
            if section_to_use is None:
                raise ValidationException(
                    "SECTION visibility requires a section to be specified",
                    field="section",
                )
        ticket.visibility = request.visibility
    if request.section is not None:
        ticket.section = request.section

    await db.commit()
    await db.refresh(ticket)

    return UpdateTicketResponse(ticket=TicketResponse.from_model(ticket))


# =============================================================================
# GET /tickets/{id}/activity
# =============================================================================


@ticket_router.get("/{ticket_id}/activity", response_model=TicketActivitiesResponse)
async def get_ticket_activity(
    ticket_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> TicketActivitiesResponse:
    """Get activity timeline for a ticket.

    Allowed only if the requester can see the ticket (same visibility rules).
    """
    # Fetch ticket with activities
    result = await db.execute(
        select(Ticket)
        .where(Ticket.id == ticket_id)
        .options(selectinload(Ticket.activities))
    )
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise NotFoundException("Ticket not found")

    # Get team_id from ticket
    team_id = ticket.team_id

    # Get user's membership (validates team access)
    membership = await _require_membership(team_id, current_user.id, db)

    # Check visibility
    if not _can_view_ticket(ticket, membership, current_user.id):
        raise ForbiddenException("You do not have access to this ticket")

    # Build response
    items = [
        TicketActivityResponse.from_model(a)
        for a in sorted(ticket.activities, key=lambda x: x.created_at)
    ]

    return TicketActivitiesResponse(items=items)


# =============================================================================
# Transition Rules (Pinned)
# =============================================================================

# Allowed transitions for ticket owner
# OPEN <-> IN_PROGRESS <-> BLOCKED
# IN_PROGRESS -> RESOLVED
# BLOCKED -> RESOLVED
ALLOWED_OWNER_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.OPEN: {TicketStatus.IN_PROGRESS},
    TicketStatus.IN_PROGRESS: {TicketStatus.OPEN, TicketStatus.BLOCKED, TicketStatus.RESOLVED},
    TicketStatus.BLOCKED: {TicketStatus.IN_PROGRESS, TicketStatus.RESOLVED},
    TicketStatus.RESOLVED: set(),  # Cannot transition from RESOLVED via /transition
    TicketStatus.VERIFIED: set(),  # Terminal state - cannot transition out
}


def _is_valid_transition(from_status: TicketStatus, to_status: TicketStatus) -> bool:
    """Check if a status transition is allowed for the owner."""
    allowed = ALLOWED_OWNER_TRANSITIONS.get(from_status, set())
    return to_status in allowed


def _can_verify_ticket(
    ticket: Ticket,
    membership: TeamMembership,
) -> bool:
    """Check if user can verify a ticket.

    Allowed: ADMIN or SECTION_LEADER in scope.
    For SECTION_LEADER, they must be in the same section as the ticket.
    """
    if membership.role == Role.ADMIN:
        return True

    if membership.role == Role.SECTION_LEADER:
        # Section leader can verify tickets in their section
        # For TEAM visibility, section leaders can verify any ticket
        if ticket.visibility == TicketVisibility.TEAM:
            return True
        # For SECTION visibility, must be the same section
        if ticket.visibility == TicketVisibility.SECTION and ticket.section == membership.section:
            return True
        # For PRIVATE, section leader can verify if ticket is in their section
        if ticket.visibility == TicketVisibility.PRIVATE and ticket.section == membership.section:
            return True

    return False


# =============================================================================
# POST /tickets/{id}/transition
# =============================================================================


@ticket_router.post("/{ticket_id}/transition", response_model=TransitionTicketResponse)
async def transition_ticket(
    ticket_id: uuid.UUID,
    request: TransitionTicketRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> TransitionTicketResponse:
    """Transition a ticket's status.

    Transition rules:
    - Owner can: OPEN <-> IN_PROGRESS <-> BLOCKED
    - Owner can: IN_PROGRESS/BLOCKED -> RESOLVED (requires content)
    - VERIFIED only via /verify endpoint (not allowed here)
    - VERIFIED is terminal (cannot transition out)

    Creates a ticket_activity row with old_status, new_status, and optional content.
    """
    # Fetch ticket
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise NotFoundException("Ticket not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(ticket.team_id, current_user.id, db)

    # Check visibility - user must be able to see the ticket
    if not _can_view_ticket(ticket, membership, current_user.id):
        raise ForbiddenException("You do not have access to this ticket")

    # Only owner can transition (RBAC rule: owner-only for /transition)
    is_owner = ticket.owner_id == current_user.id or ticket.created_by == current_user.id
    if not is_owner:
        raise ForbiddenException("Only the ticket owner can transition the status")

    current_status = ticket.status
    to_status = request.to_status

    # Cannot transition to VERIFIED via /transition endpoint
    if to_status == TicketStatus.VERIFIED:
        raise ValidationException(
            "Cannot transition to VERIFIED via this endpoint. Use /verify instead.",
            field="to_status",
        )

    # Validate the transition is allowed
    if not _is_valid_transition(current_status, to_status):
        raise ValidationException(
            f"Invalid transition from {current_status.value} to {to_status.value}",
            field="to_status",
        )

    # RESOLVED requires content
    if to_status == TicketStatus.RESOLVED and not request.content:
        raise ValidationException(
            "Transition to RESOLVED requires a note (content)",
            field="content",
        )

    # Perform transition atomically
    old_status = ticket.status
    ticket.status = to_status
    ticket.updated_at = datetime.now(datetime.now().astimezone().tzinfo)

    # If resolved, store the note and timestamp
    if to_status == TicketStatus.RESOLVED:
        ticket.resolved_note = request.content
        ticket.resolved_at = datetime.now(datetime.now().astimezone().tzinfo)

    # Create activity record
    activity = TicketActivity(
        ticket_id=ticket_id,
        user_id=current_user.id,
        type=TicketActivityType.STATUS_CHANGE,
        old_status=old_status,
        new_status=to_status,
        content=request.content,
    )
    db.add(activity)

    await db.commit()
    await db.refresh(ticket)

    return TransitionTicketResponse(ticket=TicketResponse.from_model(ticket))


# =============================================================================
# POST /tickets/{id}/verify
# =============================================================================


@ticket_router.post("/{ticket_id}/verify", response_model=VerifyTicketResponse)
async def verify_ticket(
    ticket_id: uuid.UUID,
    request: VerifyTicketRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_non_demo()),
) -> VerifyTicketResponse:
    """Verify a ticket (leadership only).

    Must be SECTION_LEADER (in scope) or ADMIN.
    Sets verified_by, verified_at, and optional verified_note.
    VERIFIED is terminal - cannot transition out.
    """
    # Fetch ticket
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise NotFoundException("Ticket not found")

    # Get user's membership (validates team access)
    membership = await _require_membership(ticket.team_id, current_user.id, db)

    # Check visibility - user must be able to see the ticket
    if not _can_view_ticket(ticket, membership, current_user.id):
        raise ForbiddenException("You do not have access to this ticket")

    # Check verify permission - only ADMIN or SECTION_LEADER in scope
    if not _can_verify_ticket(ticket, membership):
        raise ForbiddenException("Only leaders can verify tickets")

    # Cannot verify if already verified (terminal state)
    if ticket.status == TicketStatus.VERIFIED:
        raise ValidationException(
            "Ticket is already verified",
            field="status",
        )

    # Perform verification atomically
    old_status = ticket.status
    now = datetime.now(datetime.now().astimezone().tzinfo)

    ticket.status = TicketStatus.VERIFIED
    ticket.verified_by = current_user.id
    ticket.verified_at = now
    ticket.verified_note = request.content
    ticket.updated_at = now

    # Create activity record
    activity = TicketActivity(
        ticket_id=ticket_id,
        user_id=current_user.id,
        type=TicketActivityType.VERIFIED,
        old_status=old_status,
        new_status=TicketStatus.VERIFIED,
        content=request.content,
    )
    db.add(activity)

    await db.commit()
    await db.refresh(ticket)

    return VerifyTicketResponse(ticket=TicketResponse.from_model(ticket))

