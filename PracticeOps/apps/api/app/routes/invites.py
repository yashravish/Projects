"""Invite routes (public endpoints).

GET /invites/{token} - Preview invite details
POST /invites/{token}/accept - Accept invite

These endpoints are public but token validation provides security.

Pinned Invite Acceptance Flow:
1. If user is logged in:
   - Create membership using invite role/section
   - Mark used_at
   - Reject if token expired or already used

2. If user is NOT logged in:
   - If invite has email and an account exists:
     - Reject with ACCOUNT_EXISTS_LOGIN_REQUIRED
   - Otherwise:
     - Create user from { name, email, password }
     - Create membership
     - Mark used_at

Token Rules (Strict):
- Single-use: if used_at is set → reject
- Time-bound: if expires_at < now → reject
"""

import hashlib
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ConflictException, ForbiddenException, NotFoundException
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_user_id_from_token,
    hash_password,
)
from app.database import get_db
from app.models import Invite, Team, TeamMembership, User
from app.schemas.teams import (
    AcceptInviteRequest,
    AcceptInviteResponse,
    InvitePreviewResponse,
    MembershipResponse,
    RevokeInviteResponse,
)

router = APIRouter(prefix="/invites", tags=["invites"])


def hash_token(token: str) -> str:
    """Hash a token using SHA-256 for lookup.

    Must match the hash function used in teams.py for invite creation.
    """
    return hashlib.sha256(token.encode()).hexdigest()


async def _get_invite_by_token(token: str, db: AsyncSession) -> Invite | None:
    """Look up an invite by its raw token.

    The raw token is hashed and compared against stored hashes.
    """
    token_hash = hash_token(token)
    result = await db.execute(select(Invite).where(Invite.token == token_hash))
    return result.scalar_one_or_none()


async def _get_optional_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Get current user if Authorization header is present and valid.

    Returns None if no auth header or invalid token.
    This is NOT a required dependency - it's optional authentication.
    """
    if authorization is None:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    user_id = get_user_id_from_token(token, expected_type="access")

    if user_id is None:
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


@router.get("/{token}", response_model=InvitePreviewResponse)
async def preview_invite(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> InvitePreviewResponse:
    """Preview invite details.

    Public endpoint - no authentication required.
    Used by UI to show invite details before acceptance.
    """
    invite = await _get_invite_by_token(token, db)

    if invite is None:
        raise NotFoundException("Invite not found or invalid token")

    # Get team name
    team_result = await db.execute(select(Team).where(Team.id == invite.team_id))
    team = team_result.scalar_one_or_none()

    if team is None:
        raise NotFoundException("Team not found")

    # Check if expired
    is_expired = datetime.now(UTC) > invite.expires_at

    # Also mark as expired if already used
    if invite.used_at is not None:
        is_expired = True

    return InvitePreviewResponse(
        team_name=team.name,
        email=invite.email,
        role=invite.role,
        section=invite.section,
        expired=is_expired,
    )


@router.post(
    "/{token}/accept",
    response_model=AcceptInviteResponse,
    status_code=status.HTTP_201_CREATED,
)
async def accept_invite(
    token: str,
    request: AcceptInviteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(_get_optional_current_user),
) -> AcceptInviteResponse:
    """Accept a team invite.

    Pinned Invite Acceptance Flow:

    1. If user is logged in:
       - Create membership using invite role/section
       - Mark used_at
       - Reject if token expired or already used

    2. If user is NOT logged in:
       - If invite has email and an account exists:
         - Reject with ACCOUNT_EXISTS_LOGIN_REQUIRED
       - Otherwise:
         - Create user from { name, email, password }
         - Create membership
         - Mark used_at

    Token Rules (Strict):
    - Single-use: if used_at is set → reject
    - Time-bound: if expires_at < now → reject
    """
    # Find invite by token hash
    invite = await _get_invite_by_token(token, db)

    if invite is None:
        raise NotFoundException("Invite not found or invalid token")

    # Token Rule: Single-use check
    if invite.used_at is not None:
        raise ConflictException("Invite has already been used")

    # Token Rule: Time-bound check
    if datetime.now(UTC) > invite.expires_at:
        raise ForbiddenException("Invite has expired")

    # === FLOW BRANCH: Logged in user ===
    if current_user is not None:
        return await _accept_invite_logged_in(invite, current_user, db)

    # === FLOW BRANCH: Not logged in ===
    return await _accept_invite_not_logged_in(invite, request, db)


async def _accept_invite_logged_in(
    invite: Invite,
    user: User,
    db: AsyncSession,
) -> AcceptInviteResponse:
    """Handle invite acceptance for logged-in users.

    - Create membership using invite role/section
    - Mark used_at
    - Check for existing membership (avoid duplicates)
    """
    # Check if user is already a member of this team
    existing_membership = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == invite.team_id,
            TeamMembership.user_id == user.id,
        )
    )
    if existing_membership.scalar_one_or_none() is not None:
        raise ConflictException("User is already a member of this team")

    # Create membership
    membership = TeamMembership(
        team_id=invite.team_id,
        user_id=user.id,
        role=invite.role,
        section=invite.section,
    )
    db.add(membership)

    # Mark invite as used
    invite.used_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(membership)

    return AcceptInviteResponse(
        membership=MembershipResponse(
            id=membership.id,
            team_id=membership.team_id,
            user_id=membership.user_id,
            role=membership.role,
            section=membership.section,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        ),
        access_token=None,
        refresh_token=None,
    )


async def _accept_invite_not_logged_in(
    invite: Invite,
    request: AcceptInviteRequest,
    db: AsyncSession,
) -> AcceptInviteResponse:
    """Handle invite acceptance for non-logged-in users.

    Flow:
    1. If invite has email and an account exists → ACCOUNT_EXISTS_LOGIN_REQUIRED
    2. Otherwise → Create user and membership
    """
    # Determine email to use
    email = request.email or invite.email

    # If invite has email, check if account exists
    if invite.email is not None:
        existing_user = await db.execute(
            select(User).where(User.email == invite.email)
        )
        if existing_user.scalar_one_or_none() is not None:
            raise ConflictException(
                "An account with this email exists. Please log in to accept this invite."
            )

    # Validate required fields for new user
    if email is None:
        raise ConflictException(
            "Email is required to create an account. "
            "Please provide email in request or log in first."
        )

    if request.password is None:
        raise ConflictException(
            "Password is required to create an account. "
            "Please provide password in request or log in first."
        )

    if request.name is None:
        raise ConflictException(
            "Name is required to create an account. "
            "Please provide name in request or log in first."
        )

    # Check if email is already registered (for new email provided in request)
    if request.email is not None:
        existing_by_request_email = await db.execute(
            select(User).where(User.email == request.email)
        )
        if existing_by_request_email.scalar_one_or_none() is not None:
            raise ConflictException(
                "An account with this email already exists. Please log in first.",
                field="email",
            )

    # Create new user
    user = User(
        email=email,
        display_name=request.name,
        password_hash=hash_password(request.password),
    )
    db.add(user)
    await db.flush()  # Get user.id

    # Create membership
    membership = TeamMembership(
        team_id=invite.team_id,
        user_id=user.id,
        role=invite.role,
        section=invite.section,
    )
    db.add(membership)

    # Mark invite as used
    invite.used_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(user)
    await db.refresh(membership)

    # Generate auth tokens for new user
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return AcceptInviteResponse(
        membership=MembershipResponse(
            id=membership.id,
            team_id=membership.team_id,
            user_id=membership.user_id,
            role=membership.role,
            section=membership.section,
            created_at=membership.created_at,
            updated_at=membership.updated_at,
        ),
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.delete("/{invite_id}", response_model=RevokeInviteResponse)
async def revoke_invite(
    invite_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> RevokeInviteResponse:
    """Revoke an invite.

    Allowed: ADMIN of the team that the invite belongs to.
    Can only revoke unused invites.

    Note: invite_id can be either the invite UUID or the raw token.
    """
    import uuid as uuid_module

    from app.models import Role, TeamMembership

    # Try to find invite by ID first, then by token hash
    invite = None

    # Try parsing as UUID
    try:
        parsed_uuid = uuid_module.UUID(invite_id)
        result = await db.execute(select(Invite).where(Invite.id == parsed_uuid))
        invite = result.scalar_one_or_none()
    except ValueError:
        # Not a valid UUID, try as token
        token_hash = hash_token(invite_id)
        result = await db.execute(select(Invite).where(Invite.token == token_hash))
        invite = result.scalar_one_or_none()

    if invite is None:
        raise NotFoundException("Invite not found")

    # Check that current user is ADMIN of the invite's team
    membership_result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == invite.team_id,
            TeamMembership.user_id == current_user.id,
        )
    )
    membership = membership_result.scalar_one_or_none()

    if membership is None or membership.role != Role.ADMIN:
        raise ForbiddenException("Admin role required")

    # Cannot revoke an already-used invite
    if invite.used_at is not None:
        raise ConflictException("Cannot revoke an invite that has already been used")

    # Delete the invite
    await db.delete(invite)
    await db.commit()

    return RevokeInviteResponse(message="Invite revoked successfully")

