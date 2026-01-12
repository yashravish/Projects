"""Rehearsal cycle routes.

POST /teams/{team_id}/cycles - Create cycle (ADMIN, SECTION_LEADER)
GET /teams/{team_id}/cycles - List cycles with pagination (any member)
GET /teams/{team_id}/cycles/active - Get active cycle (any member)

RBAC is enforced via dependencies, not inline logic.
"""

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ConflictException, ForbiddenException
from app.database import get_db
from app.models import RehearsalCycle, Role, TeamMembership
from app.schemas.cycles import (
    ActiveCycleResponse,
    CreateCycleRequest,
    CreateCycleResponse,
    CycleResponse,
    CyclesListResponse,
)

router = APIRouter(prefix="/teams/{team_id}/cycles", tags=["cycles"])


def encode_cycle_cursor(cycle_date: datetime, id_: uuid.UUID) -> str:
    """Encode pagination cursor as base64 JSON for cycles.

    Cursor format: {"date": "ISO8601", "id": "UUID"}
    Using date as primary sort key for cycle pagination.
    """
    data = {"date": cycle_date.isoformat(), "id": str(id_)}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cycle_cursor(cursor: str) -> tuple[datetime, uuid.UUID] | None:
    """Decode pagination cursor from base64 JSON.

    Returns None if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        cycle_date = datetime.fromisoformat(data["date"])
        id_ = uuid.UUID(data["id"])
        return cycle_date, id_
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


async def _get_membership(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership | None:
    """Get user's membership for a team.

    Returns None if user is not a member.
    """
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
    """Require user to be a member of the team.

    Raises ForbiddenException if not a member.
    """
    membership = await _get_membership(team_id, user_id, db)
    if membership is None:
        raise ForbiddenException("Not a member of this team")
    return membership


async def _require_leader_or_admin(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership:
    """Require user to be ADMIN or SECTION_LEADER of the team.

    Raises ForbiddenException if not authorized.
    """
    membership = await _get_membership(team_id, user_id, db)
    if membership is None:
        raise ForbiddenException("Not a member of this team")

    if membership.role not in [Role.ADMIN, Role.SECTION_LEADER]:
        raise ForbiddenException("Admin or Section Leader role required")

    return membership


@router.post("", response_model=CreateCycleResponse, status_code=status.HTTP_201_CREATED)
async def create_cycle(
    team_id: uuid.UUID,
    request: CreateCycleRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateCycleResponse:
    """Create a new rehearsal cycle.

    Allowed roles: ADMIN, SECTION_LEADER

    Must respect DB constraint: Unique (team_id, date).
    """
    # RBAC: Require ADMIN or SECTION_LEADER
    await _require_leader_or_admin(team_id, current_user.id, db)

    # Convert date to datetime at midnight UTC
    cycle_datetime = datetime.combine(
        request.date, datetime.min.time(), tzinfo=UTC
    )

    # Generate label from date if not provided
    label = request.label
    if label is None:
        label = f"Rehearsal {request.date.strftime('%b %d, %Y')}"

    # Create cycle
    cycle = RehearsalCycle(
        team_id=team_id,
        name=label,
        date=cycle_datetime,
    )
    db.add(cycle)

    try:
        await db.commit()
        await db.refresh(cycle)
    except IntegrityError:
        await db.rollback()
        raise ConflictException(
            f"A cycle already exists for this date: {request.date}",
            field="date",
        ) from None

    return CreateCycleResponse(
        cycle=CycleResponse(
            id=cycle.id,
            team_id=cycle.team_id,
            name=cycle.name,
            date=cycle.date,
            created_at=cycle.created_at,
        )
    )


@router.get("", response_model=CyclesListResponse)
async def list_cycles(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    upcoming: Annotated[bool, Query()] = True,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> CyclesListResponse:
    """List rehearsal cycles with pagination.

    Allowed roles: Any team member

    Query params:
    - upcoming=true: Sort by date ASC (future cycles first)
    - upcoming=false: Sort by date DESC (past cycles first)

    Sorting is deterministic with id as tie-breaker.
    """
    # RBAC: Require membership (any role)
    await _require_membership(team_id, current_user.id, db)

    # Get today's date at midnight UTC for upcoming filter
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Build base query
    query = select(RehearsalCycle).where(RehearsalCycle.team_id == team_id)

    # Apply upcoming/past filter
    if upcoming:
        query = query.where(RehearsalCycle.date >= today)
    else:
        query = query.where(RehearsalCycle.date < today)

    # Apply cursor filter if provided
    if cursor:
        decoded = decode_cycle_cursor(cursor)
        if decoded:
            cursor_date, cursor_id = decoded
            if upcoming:
                # ASC ordering: get items after cursor position
                query = query.where(
                    (RehearsalCycle.date > cursor_date)
                    | (
                        (RehearsalCycle.date == cursor_date)
                        & (RehearsalCycle.id > cursor_id)
                    )
                )
            else:
                # DESC ordering: get items before cursor position
                query = query.where(
                    (RehearsalCycle.date < cursor_date)
                    | (
                        (RehearsalCycle.date == cursor_date)
                        & (RehearsalCycle.id < cursor_id)
                    )
                )

    # Apply ordering
    if upcoming:
        # Upcoming: date ASC, id ASC (earliest first)
        query = query.order_by(RehearsalCycle.date.asc(), RehearsalCycle.id.asc())
    else:
        # Past: date DESC, id DESC (most recent first)
        query = query.order_by(RehearsalCycle.date.desc(), RehearsalCycle.id.desc())

    # Fetch limit + 1 to determine if there's a next page
    query = query.limit(limit + 1)
    result = await db.execute(query)
    cycles = list(result.scalars().all())

    # Check if there's a next page
    has_next = len(cycles) > limit
    if has_next:
        cycles = cycles[:limit]

    # Build response items
    items = [
        CycleResponse(
            id=cycle.id,
            team_id=cycle.team_id,
            name=cycle.name,
            date=cycle.date,
            created_at=cycle.created_at,
        )
        for cycle in cycles
    ]

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and cycles:
        last_cycle = cycles[-1]
        next_cursor = encode_cycle_cursor(last_cycle.date, last_cycle.id)

    return CyclesListResponse(items=items, next_cursor=next_cursor)


@router.get("/active", response_model=ActiveCycleResponse)
async def get_active_cycle(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ActiveCycleResponse:
    """Get the active rehearsal cycle.

    Allowed roles: Any team member

    Active cycle selection logic:
    1. Nearest upcoming cycle (date >= today) - sorted by date ASC, id ASC
    2. Else, latest past cycle (date < today) - sorted by date DESC, id DESC
    3. Else, null if no cycles exist
    """
    # RBAC: Require membership (any role)
    await _require_membership(team_id, current_user.id, db)

    # Get today's date at midnight UTC
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Step 1: Try to find nearest upcoming cycle (date >= today)
    upcoming_query = (
        select(RehearsalCycle)
        .where(
            RehearsalCycle.team_id == team_id,
            RehearsalCycle.date >= today,
        )
        .order_by(RehearsalCycle.date.asc(), RehearsalCycle.id.asc())
        .limit(1)
    )
    result = await db.execute(upcoming_query)
    cycle = result.scalar_one_or_none()

    # Step 2: If no upcoming, try to find latest past cycle (date < today)
    if cycle is None:
        past_query = (
            select(RehearsalCycle)
            .where(
                RehearsalCycle.team_id == team_id,
                RehearsalCycle.date < today,
            )
            .order_by(RehearsalCycle.date.desc(), RehearsalCycle.id.desc())
            .limit(1)
        )
        result = await db.execute(past_query)
        cycle = result.scalar_one_or_none()

    # Step 3: Return cycle or null
    if cycle is None:
        return ActiveCycleResponse(cycle=None)

    return ActiveCycleResponse(
        cycle=CycleResponse(
            id=cycle.id,
            team_id=cycle.team_id,
            name=cycle.name,
            date=cycle.date,
            created_at=cycle.created_at,
        )
    )

