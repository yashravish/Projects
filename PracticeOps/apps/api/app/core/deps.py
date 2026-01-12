"""FastAPI dependencies for authentication and RBAC.

Implements:
- require_auth() - Validates JWT and returns current user
- require_membership(team_id) - Ensures user is member of team
- require_role(team_id, roles) - Ensures user has required role in team
- require_section_leader_of_section(team_id, section) - Ensures user is section leader of specific section
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import DemoReadOnlyException, ForbiddenException, UnauthorizedException
from app.core.security import get_user_id_from_token
from app.database import get_db
from app.models import Role, TeamMembership, User


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from the Authorization header.

    Expects: Authorization: Bearer <token>
    """
    if authorization is None:
        raise UnauthorizedException("Missing authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise UnauthorizedException("Invalid authorization header format")

    token = parts[1]
    user_id = get_user_id_from_token(token, expected_type="access")

    if user_id is None:
        raise UnauthorizedException("Invalid or expired token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise UnauthorizedException("User not found")

    return user


# Type alias for the current user dependency
CurrentUser = Annotated[User, Depends(get_current_user)]


def require_auth() -> Callable[[User], User]:
    """Dependency that requires authentication.

    Returns the current user if authenticated.
    """

    def dependency(current_user: CurrentUser) -> User:
        return current_user

    return dependency


def require_membership(
    team_id: uuid.UUID,
) -> Callable[..., Awaitable[TeamMembership]]:
    """Dependency that requires the user to be a member of the specified team.

    Returns the membership if found.
    """

    async def dependency(
        current_user: CurrentUser,
        db: AsyncSession = Depends(get_db),
    ) -> TeamMembership:
        result = await db.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team_id,
                TeamMembership.user_id == current_user.id,
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None:
            raise ForbiddenException("Not a member of this team")

        return membership

    return dependency


def require_role(
    team_id: uuid.UUID, roles: list[Role]
) -> Callable[..., Awaitable[TeamMembership]]:
    """Dependency that requires the user to have one of the specified roles in the team.

    Returns the membership if the role matches.
    """

    async def dependency(
        current_user: CurrentUser,
        db: AsyncSession = Depends(get_db),
    ) -> TeamMembership:
        result = await db.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team_id,
                TeamMembership.user_id == current_user.id,
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None:
            raise ForbiddenException("Not a member of this team")

        if membership.role not in roles:
            raise ForbiddenException(f"Required role: {', '.join(r.value for r in roles)}")

        return membership

    return dependency


def require_section_leader_of_section(
    team_id: uuid.UUID, section: str
) -> Callable[..., Awaitable[TeamMembership]]:
    """Dependency that requires the user to be section leader of a specific section.

    Also allows ADMIN role to pass.
    Returns the membership if authorized.
    """

    async def dependency(
        current_user: CurrentUser,
        db: AsyncSession = Depends(get_db),
    ) -> TeamMembership:
        result = await db.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team_id,
                TeamMembership.user_id == current_user.id,
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None:
            raise ForbiddenException("Not a member of this team")

        # Admins can access any section
        if membership.role == Role.ADMIN:
            return membership

        # Section leaders can only access their own section
        if membership.role == Role.SECTION_LEADER:
            if membership.section == section:
                return membership
            raise ForbiddenException(f"Not section leader of {section}")

        # Members cannot access section leader features
        raise ForbiddenException("Section leader or admin role required")

    return dependency


def is_demo_user(user: User) -> bool:
    """Check if user is a demo account based on email domain.

    Demo users have email addresses ending with @practiceops.app
    """
    return user.email.endswith("@practiceops.app")


def require_non_demo() -> Callable[[User], User]:
    """Dependency that blocks write operations from demo users.

    Demo accounts are read-only for data integrity and security.
    Returns the current user if not a demo account.
    Raises DemoReadOnlyException if user is a demo account.
    """

    def dependency(current_user: CurrentUser) -> User:
        if is_demo_user(current_user):
            raise DemoReadOnlyException()
        return current_user

    return dependency
