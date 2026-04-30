"""Data access object for users."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    """Encapsulates queries against the users table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, user_id: int) -> Optional[User]:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email.lower().strip())
        return self.db.scalars(stmt).first()

    def list_all(self) -> List[User]:
        stmt = select(User).order_by(User.name)
        return list(self.db.scalars(stmt).all())

    def add(self, user: User) -> User:
        self.db.add(user)
        self.db.flush()
        return user
