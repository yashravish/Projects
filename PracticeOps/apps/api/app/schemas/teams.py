"""Team, membership, and invite schemas for request/response models.

Implements contracts for:
- POST /teams
- GET /teams/{team_id}/members
- PATCH /teams/{team_id}/members/{user_id}
- POST /teams/{team_id}/invites
- GET /invites/{token}
- POST /invites/{token}/accept
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import Role

# =============================================================================
# Team Schemas
# =============================================================================


class CreateTeamRequest(BaseModel):
    """Request schema for creating a team."""

    name: str = Field(min_length=1, max_length=100)


class TeamResponse(BaseModel):
    """Team data in responses."""

    id: uuid.UUID
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateTeamResponse(BaseModel):
    """Response schema for team creation."""

    team: TeamResponse


class UpdateTeamRequest(BaseModel):
    """Request schema for updating a team."""

    name: str = Field(min_length=1, max_length=100)


class UpdateTeamResponse(BaseModel):
    """Response schema for team update."""

    team: TeamResponse


# =============================================================================
# Membership Schemas
# =============================================================================


class MemberResponse(BaseModel):
    """Member data in list responses."""

    id: uuid.UUID
    user_id: uuid.UUID
    email: str
    display_name: str
    role: Role
    section: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MembersListResponse(BaseModel):
    """Paginated list of team members.

    Follows pagination contract from systemprompt.md.
    """

    items: list[MemberResponse]
    next_cursor: str | None


class UpdateMemberRequest(BaseModel):
    """Request schema for updating a team membership."""

    role: Role | None = None
    section: str | None = Field(default=None, max_length=50)
    primary_team: bool | None = None


class MembershipResponse(BaseModel):
    """Single membership data in responses."""

    id: uuid.UUID
    team_id: uuid.UUID
    user_id: uuid.UUID
    role: Role
    section: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UpdateMemberResponse(BaseModel):
    """Response schema for membership update."""

    membership: MembershipResponse


# =============================================================================
# Invite Schemas
# =============================================================================


class CreateInviteRequest(BaseModel):
    """Request schema for creating a team invite.

    Token rules:
    - Token is returned only once
    - Raw token must never be persisted
    """

    email: EmailStr | None = None
    role: Role = Role.MEMBER
    section: str | None = Field(default=None, max_length=50)
    expires_in_hours: int = Field(default=168, ge=1, le=720)  # 1 hour to 30 days


class CreateInviteResponse(BaseModel):
    """Response schema for invite creation.

    Security: invite_link contains the raw token which is returned ONLY ONCE.
    The token is hashed before storage in the database.
    """

    invite_link: str


class InvitePreviewResponse(BaseModel):
    """Response schema for invite preview (public endpoint).

    Used by UI to show invite details before acceptance.
    """

    team_name: str
    email: str | None
    role: Role
    section: str | None
    expired: bool


class AcceptInviteRequest(BaseModel):
    """Request schema for accepting an invite when not logged in.

    Only required when creating a new account during invite acceptance.
    """

    name: str | None = Field(default=None, min_length=1, max_length=100)
    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)


class AcceptInviteResponse(BaseModel):
    """Response schema for accepting an invite.

    If a new user was created, includes auth tokens.
    If existing user accepted, only includes membership.
    """

    membership: MembershipResponse
    access_token: str | None = None
    refresh_token: str | None = None


class InviteResponse(BaseModel):
    """Invite data for list responses.

    Note: Token is NOT included as it's only returned once at creation.
    """

    id: uuid.UUID
    email: str | None
    role: Role
    section: str | None
    expires_at: datetime
    used_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class InvitesListResponse(BaseModel):
    """Paginated list of team invites.

    Follows pagination contract from systemprompt.md.
    """

    items: list[InviteResponse]
    next_cursor: str | None


class RevokeInviteResponse(BaseModel):
    """Response schema for revoking an invite."""

    message: str = "Invite revoked successfully"

