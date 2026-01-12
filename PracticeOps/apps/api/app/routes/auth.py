"""Authentication routes.

POST /auth/register - Register new user
POST /auth/login - Login and get tokens
POST /auth/refresh - Refresh access token
GET /me - Get current user info

Rate limited: /auth/login and /auth/register are limited to 10 requests/minute/IP
"""

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ConflictException, UnauthorizedException
from app.core.middleware import AUTH_RATE_LIMIT, limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_user_id_from_token,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.models import TeamMembership, User
from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    MeResponse,
    RefreshRequest,
    RefreshResponse,
    RegisterRequest,
    TeamMembershipResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(AUTH_RATE_LIMIT)
async def register(
    request: Request,  # Required for rate limiter - must be named 'request'
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Register a new user."""
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == body.email))
    existing_user = result.scalar_one_or_none()

    if existing_user is not None:
        raise ConflictException("Email already registered", field="email")

    # Create new user
    user = User(
        email=body.email,
        display_name=body.name,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.display_name,
        ),
    )


@router.post("/login", response_model=AuthResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def login(
    request: Request,  # Required for rate limiter - must be named 'request'
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Login with email and password."""
    # Find user by email
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None:
        raise UnauthorizedException("Invalid email or password")

    # Verify password
    if not verify_password(body.password, user.password_hash):
        raise UnauthorizedException("Invalid email or password")

    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.display_name,
        ),
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> RefreshResponse:
    """Refresh access token using refresh token."""
    # Validate refresh token
    user_id = get_user_id_from_token(body.refresh_token, expected_type="refresh")

    if user_id is None:
        raise UnauthorizedException("Invalid or expired refresh token")

    # Verify user still exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise UnauthorizedException("User not found")

    # Generate new access token
    access_token = create_access_token(user.id)

    return RefreshResponse(access_token=access_token)


# Separate router for /me to avoid prefix
me_router = APIRouter(tags=["auth"])


@me_router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> MeResponse:
    """Get current user info with primary team membership."""
    # Get user's first team membership (primary team)
    result = await db.execute(
        select(TeamMembership)
        .where(TeamMembership.user_id == current_user.id)
        .order_by(TeamMembership.created_at)
        .limit(1)
    )
    membership = result.scalar_one_or_none()

    primary_team = None
    if membership is not None:
        primary_team = TeamMembershipResponse(
            team_id=membership.team_id,
            role=membership.role,
            section=membership.section,
        )

    return MeResponse(
        user=UserResponse(
            id=current_user.id,
            email=current_user.email,
            name=current_user.display_name,
        ),
        primary_team=primary_team,
    )

