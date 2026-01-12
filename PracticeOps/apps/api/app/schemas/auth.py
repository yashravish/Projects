"""Authentication schemas for request/response models."""

import uuid

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import Role


class RegisterRequest(BaseModel):
    """Request schema for user registration."""

    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request schema for user login."""

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Request schema for token refresh."""

    refresh_token: str


class UserResponse(BaseModel):
    """User data in responses."""

    id: uuid.UUID
    email: str
    name: str

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """Response schema for register and login."""

    access_token: str
    refresh_token: str
    user: UserResponse


class RefreshResponse(BaseModel):
    """Response schema for token refresh."""

    access_token: str


class TeamMembershipResponse(BaseModel):
    """Team membership data for /me response."""

    team_id: uuid.UUID
    role: Role
    section: str | None

    model_config = {"from_attributes": True}


class MeResponse(BaseModel):
    """Response schema for GET /me."""

    user: UserResponse
    primary_team: TeamMembershipResponse | None

