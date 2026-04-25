"""JWT issuing and verification.

Supports HS256 (with `JWT_SECRET`) and RS256 (with key files at
`JWT_PRIVATE_KEY_PATH` / `JWT_PUBLIC_KEY_PATH`). Selection happens
automatically via `Settings.jwt_algorithm`.

Tokens carry: `sub` (user id), `org` (organization id), `role`, `type`
(access|refresh), `iss`, `iat`, `exp`, `jti`.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any, Literal, cast

from jose import JWTError, jwt

from app.config import Settings, get_settings


class TokenError(Exception):
    """Raised when a token cannot be decoded or has expired."""


def _load_keys(settings: Settings) -> tuple[str, str]:
    """Return `(signing_key, verification_key)`."""
    if settings.jwt_uses_rs256:
        priv = Path(settings.jwt_private_key_path).read_text(encoding="utf-8")
        pub = (
            Path(settings.jwt_public_key_path).read_text(encoding="utf-8")
            if settings.jwt_public_key_path
            else priv
        )
        return priv, pub
    return settings.jwt_secret, settings.jwt_secret


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _build_payload(
    *,
    user_id: uuid.UUID,
    organization_id: uuid.UUID,
    role: str,
    token_type: Literal["access", "refresh"],
    ttl: dt.timedelta,
    issuer: str,
) -> dict[str, Any]:
    now = _now()
    return {
        "sub": str(user_id),
        "org": str(organization_id),
        "role": role,
        "type": token_type,
        "iss": issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "jti": uuid.uuid4().hex,
    }


def create_access_token(
    *, user_id: uuid.UUID, organization_id: uuid.UUID, role: str
) -> str:
    settings = get_settings()
    signing_key, _ = _load_keys(settings)
    payload = _build_payload(
        user_id=user_id,
        organization_id=organization_id,
        role=role,
        token_type="access",
        ttl=dt.timedelta(minutes=settings.jwt_access_ttl_minutes),
        issuer=settings.jwt_issuer,
    )
    return cast(str, jwt.encode(payload, signing_key, algorithm=settings.jwt_algorithm))


def create_refresh_token(
    *, user_id: uuid.UUID, organization_id: uuid.UUID, role: str
) -> str:
    settings = get_settings()
    signing_key, _ = _load_keys(settings)
    payload = _build_payload(
        user_id=user_id,
        organization_id=organization_id,
        role=role,
        token_type="refresh",
        ttl=dt.timedelta(days=settings.jwt_refresh_ttl_days),
        issuer=settings.jwt_issuer,
    )
    return cast(str, jwt.encode(payload, signing_key, algorithm=settings.jwt_algorithm))


def decode_token(token: str, *, expected_type: Literal["access", "refresh"]) -> dict[str, Any]:
    settings = get_settings()
    _, verification_key = _load_keys(settings)
    try:
        payload = jwt.decode(
            token,
            verification_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
        )
    except JWTError as exc:
        raise TokenError(str(exc)) from exc

    if payload.get("type") != expected_type:
        raise TokenError(f"expected token type {expected_type}, got {payload.get('type')}")
    return cast(dict[str, Any], payload)
