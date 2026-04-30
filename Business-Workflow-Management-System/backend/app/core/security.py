"""Security utilities for password hashing and JWT handling."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, TypedDict, cast

import bcrypt
from jose import JWTError, jwt

from app.core.config import settings


# bcrypt has a hard 72-byte limit on inputs. We truncate to that length to
# match common library behavior and avoid surprising errors at runtime.
_BCRYPT_MAX_BYTES = 72


class TokenPayload(TypedDict, total=False):
    """Decoded JWT claims emitted and consumed by this service."""

    sub: str
    exp: int
    iat: int
    role: str
    email: str


def _to_bcrypt_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """Return a bcrypt hash (utf-8 string) for the given plain-text password."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(_to_bcrypt_bytes(password), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            _to_bcrypt_bytes(plain_password),
            hashed_password.encode("utf-8"),
        )
    except ValueError:
        return False


def create_access_token(
    subject: str | int,
    role: str,
    email: str,
    expires_minutes: Optional[int] = None,
) -> str:
    expire_minutes = expires_minutes or settings.JWT_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload: TokenPayload = {
        "sub": str(subject),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
        "iat": int(now.timestamp()),
        "role": role,
        "email": email,
    }
    return jwt.encode(
        cast(dict, payload), settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def decode_access_token(token: str) -> TokenPayload:
    """Decode a JWT token, raising JWTError on failure."""
    return cast(
        TokenPayload,
        jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]),
    )


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "JWTError",
    "TokenPayload",
]
