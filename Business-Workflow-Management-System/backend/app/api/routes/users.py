"""User listing endpoints (read-only, used by the frontend assignee dropdown)."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserSummary


router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=List[UserSummary])
def list_users(
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> List[UserSummary]:
    repo = UserRepository(db)
    return [UserSummary.model_validate(u) for u in repo.list_all()]
