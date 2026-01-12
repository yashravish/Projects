"""Security utilities for password hashing and JWT tokens.

Uses:
- passlib with bcrypt for password hashing
- PyJWT for token generation and validation
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import jwt
from passlib.context import CryptContext

from app.config import settings

# Password hashing context using bcrypt
pwd_context: CryptContext = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return cast(str, pwd_context.hash(password))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return cast(bool, pwd_context.verify(plain_password, hashed_password))


def create_access_token(user_id: uuid.UUID) -> str:
    """Create a short-lived access token (15 minutes)."""
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: uuid.UUID) -> str:
    """Create a long-lived refresh token (30 days)."""
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token.

    Returns the payload if valid, None if invalid or expired.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_user_id_from_token(token: str, expected_type: str = "access") -> uuid.UUID | None:
    """Extract user ID from a valid token of the expected type.

    Returns None if token is invalid, expired, or wrong type.
    """
    payload = decode_token(token)
    if payload is None:
        return None

    if payload.get("type") != expected_type:
        return None

    sub = payload.get("sub")
    if sub is None:
        return None

    try:
        return uuid.UUID(sub)
    except ValueError:
        return None
