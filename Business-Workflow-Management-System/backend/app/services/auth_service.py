"""Authentication service - login + current user lookup."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.core.security import (
    JWTError,
    create_access_token,
    decode_access_token,
    verify_password,
)
from app.models.audit_log import AuditAction
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.audit_log_service import AuditLogService


logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.users = UserRepository(db)
        self.audit = AuditLogService(db)

    def authenticate(self, email: str, password: str) -> User:
        user = self.users.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            logger.warning("Failed login attempt for email=%s", email)
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )
        return user

    def login(self, email: str, password: str) -> tuple[str, User]:
        user = self.authenticate(email=email, password=password)
        token = create_access_token(
            subject=user.id,
            role=user.role.value,
            email=user.email,
        )
        self.audit.log_action(
            user_id=user.id,
            action=AuditAction.LOGIN,
            entity_type="user",
            entity_id=user.id,
            details=f"User {user.email} logged in",
        )
        self.db.commit()
        return token, user

    def get_user_from_token(self, token: str) -> User:
        try:
            payload = decode_access_token(token)
            subject: Optional[str] = payload.get("sub")
            if subject is None:
                raise ValueError("Missing subject claim")
            user_id = int(subject)
        except (JWTError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials.",
            ) from exc

        user = self.users.get(user_id)
        if user is None:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="User not found.",
            )
        return user
