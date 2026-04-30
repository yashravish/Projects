"""Common FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status as http_status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User, UserRole
from app.services.auth_service import AuthService


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user from the Authorization Bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    auth = AuthService(db)
    return auth.get_user_from_token(credentials.credentials)


def require_manager(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.MANAGER:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Manager role required.",
        )
    return user
