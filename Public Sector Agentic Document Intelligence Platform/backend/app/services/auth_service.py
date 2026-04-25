"""Authentication and registration business logic."""
from __future__ import annotations

import re
import unicodedata
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Organization, User
from app.security import jwt as jwt_module
from app.security.passwords import hash_password, verify_password


class AuthError(Exception):
    """Authentication / registration failure with a user-safe message."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", normalized.lower()).strip("-")
    return slug or "org"


async def register_user(
    session: AsyncSession,
    *,
    email: str,
    password: str,
    organization_name: str,
) -> tuple[User, Organization, str, str]:
    """Create a new organization + admin user atomically.

    Returns `(user, organization, access_token, refresh_token)`.
    """
    email_normalized = email.strip().lower()

    existing_user = await session.scalar(
        select(User).where(User.email == email_normalized)
    )
    if existing_user is not None:
        raise AuthError("an account with that email already exists", status_code=409)

    base_slug = _slugify(organization_name)
    slug = base_slug
    suffix = 1
    while await session.scalar(select(Organization).where(Organization.slug == slug)):
        suffix += 1
        slug = f"{base_slug}-{suffix}"
        if suffix > 1000:  # pragma: no cover - defensive
            raise AuthError("could not allocate organization slug", status_code=500)

    organization = Organization(name=organization_name.strip(), slug=slug)
    session.add(organization)
    await session.flush()

    user = User(
        organization_id=organization.id,
        email=email_normalized,
        password_hash=hash_password(password),
        role="admin",
        is_active=True,
    )
    session.add(user)

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise AuthError("registration conflict; try again", status_code=409) from exc

    await session.refresh(user)
    await session.refresh(organization)

    access = jwt_module.create_access_token(
        user_id=user.id, organization_id=organization.id, role=user.role
    )
    refresh = jwt_module.create_refresh_token(
        user_id=user.id, organization_id=organization.id, role=user.role
    )
    return user, organization, access, refresh


async def authenticate(
    session: AsyncSession, *, email: str, password: str
) -> tuple[User, str, str]:
    email_normalized = email.strip().lower()
    user = await session.scalar(select(User).where(User.email == email_normalized))
    if user is None or not user.is_active:
        raise AuthError("invalid email or password", status_code=401)
    if not verify_password(password, user.password_hash):
        raise AuthError("invalid email or password", status_code=401)

    access = jwt_module.create_access_token(
        user_id=user.id, organization_id=user.organization_id, role=user.role
    )
    refresh = jwt_module.create_refresh_token(
        user_id=user.id, organization_id=user.organization_id, role=user.role
    )
    return user, access, refresh


async def refresh_access_token(
    session: AsyncSession, *, refresh_token: str
) -> str:
    try:
        payload = jwt_module.decode_token(refresh_token, expected_type="refresh")
    except jwt_module.TokenError as exc:
        raise AuthError("invalid or expired refresh token", status_code=401) from exc

    user_id = uuid.UUID(payload["sub"])
    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthError("user not found or inactive", status_code=401)

    return jwt_module.create_access_token(
        user_id=user.id, organization_id=user.organization_id, role=user.role
    )


async def get_current_user(session: AsyncSession, *, user_id: uuid.UUID) -> User:
    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthError("user not found or inactive", status_code=401)
    return user
