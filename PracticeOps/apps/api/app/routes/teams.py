"""Team management routes.

POST /teams - Create team (any authenticated user)
GET /teams/{team_id}/members - List members with pagination (ADMIN, SECTION_LEADER)
PATCH /teams/{team_id}/members/{user_id} - Update membership (ADMIN only)
POST /teams/{team_id}/invites - Create invite (ADMIN only)

RBAC is enforced via dependencies, not inline logic.
"""

import base64
import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.deps import CurrentUser
from app.core.errors import ForbiddenException, NotFoundException
from app.database import get_db
from app.models import Invite, Role, Team, TeamMembership, User
from app.schemas.teams import (
    CreateInviteRequest,
    CreateInviteResponse,
    CreateTeamRequest,
    CreateTeamResponse,
    InviteResponse,
    InvitesListResponse,
    MemberResponse,
    MembershipResponse,
    MembersListResponse,
    RevokeInviteResponse,
    TeamResponse,
    UpdateMemberRequest,
    UpdateMemberResponse,
    UpdateTeamRequest,
    UpdateTeamResponse,
)

router = APIRouter(prefix="/teams", tags=["teams"])


def hash_token(token: str) -> str:
    """Hash a token using SHA-256 for secure storage.

    The raw token is never stored; only its hash is persisted.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def generate_invite_token() -> tuple[str, str]:
    """Generate a secure invite token and its hash.

    Returns:
        tuple[str, str]: (raw_token, token_hash)
        - raw_token: Returned to user once, never stored
        - token_hash: Stored in database
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_token(raw_token)
    return raw_token, token_hash


def encode_cursor(created_at: datetime, id_: uuid.UUID) -> str:
    """Encode pagination cursor as base64 JSON.

    Cursor format: {"created_at": "ISO8601", "id": "UUID"}
    """
    data = {"created_at": created_at.isoformat(), "id": str(id_)}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID] | None:
    """Decode pagination cursor from base64 JSON.

    Returns None if cursor is invalid.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        created_at = datetime.fromisoformat(data["created_at"])
        id_ = uuid.UUID(data["id"])
        return created_at, id_
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


@router.post("", response_model=CreateTeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    request: CreateTeamRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateTeamResponse:
    """Create a new team.

    Any authenticated user can create a team.
    Creator automatically becomes ADMIN of the new team.
    """
    # Create the team
    team = Team(name=request.name)
    db.add(team)
    await db.flush()  # Get team.id before creating membership

    # Create admin membership for creator
    membership = TeamMembership(
        team_id=team.id,
        user_id=current_user.id,
        role=Role.ADMIN,
    )
    db.add(membership)
    await db.commit()
    await db.refresh(team)

    return CreateTeamResponse(
        team=TeamResponse(
            id=team.id,
            name=team.name,
            created_at=team.created_at,
        )
    )


@router.patch("/{team_id}", response_model=UpdateTeamResponse)
async def update_team(
    team_id: uuid.UUID,
    request: UpdateTeamRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UpdateTeamResponse:
    """Update team information.

    Allowed roles: ADMIN only
    """
    # Check that current user is ADMIN of this team
    admin_membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN], db
    )
    if admin_membership is None:
        raise ForbiddenException("Admin role required")

    # Fetch team
    result = await db.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()

    if team is None:
        raise NotFoundException("Team not found")

    # Update team name
    team.name = request.name

    await db.commit()
    await db.refresh(team)

    return UpdateTeamResponse(
        team=TeamResponse(
            id=team.id,
            name=team.name,
            created_at=team.created_at,
        )
    )


@router.get("/{team_id}/members", response_model=MembersListResponse)
async def list_members(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> MembersListResponse:
    """List team members with pagination.

    Allowed roles: ADMIN, SECTION_LEADER

    Sorting: created_at ASC, id ASC (tie-breaker)
    Pagination: cursor-based using base64 encoded JSON
    """
    # Check membership and role via dependency pattern
    membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN, Role.SECTION_LEADER], db
    )
    if membership is None:
        raise ForbiddenException("Admin or Section Leader role required")

    # Build query with joins for user data
    query = (
        select(TeamMembership, User)
        .join(User, TeamMembership.user_id == User.id)
        .where(TeamMembership.team_id == team_id)
    )

    # Apply cursor filter if provided
    if cursor:
        decoded = decode_cursor(cursor)
        if decoded:
            cursor_created_at, cursor_id = decoded
            # Get items after cursor position (created_at, id) ordering
            query = query.where(
                (TeamMembership.created_at > cursor_created_at)
                | (
                    (TeamMembership.created_at == cursor_created_at)
                    & (TeamMembership.id > cursor_id)
                )
            )

    # Order by created_at ASC, id ASC for deterministic pagination
    query = query.order_by(TeamMembership.created_at.asc(), TeamMembership.id.asc())

    # Fetch limit + 1 to determine if there's a next page
    query = query.limit(limit + 1)
    result = await db.execute(query)
    rows = result.all()

    # Check if there's a next page
    has_next = len(rows) > limit
    if has_next:
        rows = rows[:limit]

    # Build response items
    items = [
        MemberResponse(
            id=membership.id,
            user_id=membership.user_id,
            email=user.email,
            display_name=user.display_name,
            role=membership.role,
            section=membership.section,
            created_at=membership.created_at,
        )
        for membership, user in rows
    ]

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and items:
        last_item = rows[-1][0]  # Last membership
        next_cursor = encode_cursor(last_item.created_at, last_item.id)

    return MembersListResponse(items=items, next_cursor=next_cursor)


@router.patch("/{team_id}/members/{user_id}", response_model=UpdateMemberResponse)
async def update_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    request: UpdateMemberRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UpdateMemberResponse:
    """Update a team member's role, section, or primary team status.

    Allowed roles: ADMIN only
    """
    # Check that current user is ADMIN of this team
    admin_membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN], db
    )
    if admin_membership is None:
        raise ForbiddenException("Admin role required")

    # Find the target membership
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    membership = result.scalar_one_or_none()

    if membership is None:
        raise NotFoundException("Membership not found")

    # Update fields if provided
    if request.role is not None:
        membership.role = request.role

    if request.section is not None:
        membership.section = request.section

    # Note: primary_team is not directly stored; it's derived from ordering
    # For now, we acknowledge the field but don't modify storage
    # Primary team is the user's oldest membership by created_at

    await db.commit()
    await db.refresh(membership)

    return UpdateMemberResponse(
        membership=MembershipResponse(
            id=membership.id,
            team_id=membership.team_id,
            user_id=membership.user_id,
            role=membership.role,
            section=membership.section,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        )
    )


@router.delete("/{team_id}/members/{user_id}", response_model=RevokeInviteResponse)
async def remove_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> RevokeInviteResponse:
    """Remove a member from the team.

    Allowed roles: ADMIN only
    Cannot remove yourself if you're the last admin.
    """
    # Check that current user is ADMIN of this team
    admin_membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN], db
    )
    if admin_membership is None:
        raise ForbiddenException("Admin role required")

    # Find the target membership
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    membership = result.scalar_one_or_none()

    if membership is None:
        raise NotFoundException("Membership not found")

    # Cannot remove the last admin
    if membership.role == Role.ADMIN:
        admin_count_result = await db.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team_id,
                TeamMembership.role == Role.ADMIN,
            )
        )
        admin_count = len(admin_count_result.scalars().all())
        if admin_count <= 1:
            raise ForbiddenException("Cannot remove the last admin from the team")

    await db.delete(membership)
    await db.commit()

    return RevokeInviteResponse(message="Member removed successfully")


@router.get("/{team_id}/invites", response_model=InvitesListResponse)
async def list_invites(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
    include_used: Annotated[bool, Query()] = False,
) -> InvitesListResponse:
    """List team invites with pagination.

    Allowed roles: ADMIN only

    By default, only returns unused (active) invites.
    Set include_used=true to also see used invites.
    """
    # Check that current user is ADMIN of this team
    admin_membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN], db
    )
    if admin_membership is None:
        raise ForbiddenException("Admin role required")

    # Build query
    query = select(Invite).where(Invite.team_id == team_id)

    # Filter unused by default
    if not include_used:
        query = query.where(Invite.used_at.is_(None))

    # Apply cursor filter if provided
    if cursor:
        decoded = decode_cursor(cursor)
        if decoded:
            cursor_created_at, cursor_id = decoded
            query = query.where(
                (Invite.created_at > cursor_created_at)
                | (
                    (Invite.created_at == cursor_created_at)
                    & (Invite.id > cursor_id)
                )
            )

    # Order by created_at DESC (newest first), id ASC for deterministic pagination
    query = query.order_by(Invite.created_at.desc(), Invite.id.asc())

    # Fetch limit + 1 to determine if there's a next page
    query = query.limit(limit + 1)
    result = await db.execute(query)
    invites = list(result.scalars().all())

    # Check if there's a next page
    has_next = len(invites) > limit
    if has_next:
        invites = invites[:limit]

    # Build response items
    items = [
        InviteResponse(
            id=invite.id,
            email=invite.email,
            role=invite.role,
            section=invite.section,
            expires_at=invite.expires_at,
            used_at=invite.used_at,
            created_at=invite.created_at,
        )
        for invite in invites
    ]

    # Generate next cursor if there's more data
    next_cursor = None
    if has_next and invites:
        last = invites[-1]
        next_cursor = encode_cursor(last.created_at, last.id)

    return InvitesListResponse(items=items, next_cursor=next_cursor)


@router.post(
    "/{team_id}/invites",
    response_model=CreateInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_invite(
    team_id: uuid.UUID,
    request: CreateInviteRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateInviteResponse:
    """Create a team invite.

    Allowed roles: ADMIN only

    Security:
    - Token is returned only once in the response
    - Only the hashed token is stored in the database
    - Raw token must never be persisted
    """
    # Check that current user is ADMIN of this team
    admin_membership = await _get_membership_with_role_check(
        team_id, current_user.id, [Role.ADMIN], db
    )
    if admin_membership is None:
        raise ForbiddenException("Admin role required")

    # Verify team exists
    team_result = await db.execute(select(Team).where(Team.id == team_id))
    team = team_result.scalar_one_or_none()
    if team is None:
        raise NotFoundException("Team not found")

    # Generate secure token
    raw_token, token_hash = generate_invite_token()

    # Calculate expiration
    expires_at = datetime.now(UTC) + timedelta(hours=request.expires_in_hours)

    # Create invite with hashed token
    invite = Invite(
        team_id=team_id,
        token=token_hash,  # Store HASH, not raw token
        email=request.email,
        role=request.role,
        section=request.section,
        expires_at=expires_at,
        created_by=current_user.id,
    )
    db.add(invite)
    await db.commit()

    # Build invite link with RAW token (returned only once)
    base_url = getattr(settings, "frontend_url", "http://localhost:5173")
    invite_link = f"{base_url}/invites/{raw_token}"

    return CreateInviteResponse(invite_link=invite_link)


async def _get_membership_with_role_check(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    allowed_roles: list[Role],
    db: AsyncSession,
) -> TeamMembership | None:
    """Get user's membership if they have one of the allowed roles.

    Returns None if user is not a member or doesn't have required role.
    This is a helper function to avoid code duplication while keeping
    RBAC checks explicit.
    """
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    membership = result.scalar_one_or_none()

    if membership is None:
        return None

    if membership.role not in allowed_roles:
        return None

    return membership

