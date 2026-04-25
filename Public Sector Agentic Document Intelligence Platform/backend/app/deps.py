"""FastAPI dependencies shared across routes."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import get_session
from app.security import jwt as jwt_module
from app.services import auth_service

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def _extract_bearer_token(
    authorization: str | None = Header(default=None),
) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return authorization.split(" ", 1)[1].strip()


async def get_current_user(
    session: SessionDep,
    token: str = Depends(_extract_bearer_token),
) -> User:
    try:
        payload = jwt_module.decode_token(token, expected_type="access")
    except jwt_module.TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="malformed token"
        ) from exc

    try:
        return await auth_service.get_current_user(session, user_id=user_id)
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: str):  # type: ignore[no-untyped-def]
    """Return a dependency that ensures the current user has one of `roles`."""

    async def _checker(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{user.role}' is not permitted; requires {sorted(roles)}",
            )
        return user

    return _checker
