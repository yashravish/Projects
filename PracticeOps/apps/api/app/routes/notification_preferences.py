"""Notification preferences routes.

Allows users to view and update their own notification preferences.
Users can only access their own preferences, not those of other users.
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ForbiddenException
from app.database import get_db
from app.models import NotificationPreference, TeamMembership
from app.schemas.notification_preferences import (
    NotificationPreferencesResponse,
    NotificationPreferencesUpdate,
)

router = APIRouter(prefix="/teams/{team_id}/notification-preferences", tags=["notifications"])


async def get_or_create_preferences(
    db: AsyncSession,
    user_id: uuid.UUID,
    team_id: uuid.UUID,
) -> NotificationPreference:
    """Get user's notification preferences, creating if not exist."""
    result = await db.execute(
        select(NotificationPreference).where(
            NotificationPreference.user_id == user_id,
            NotificationPreference.team_id == team_id,
        )
    )
    prefs = result.scalar_one_or_none()

    if prefs is None:
        # Create default preferences
        prefs = NotificationPreference(
            user_id=user_id,
            team_id=team_id,
        )
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)

    return prefs


@router.get("", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferencesResponse:
    """Get current user's notification preferences for a team.

    Returns the user's preferences, creating default ones if they don't exist.
    """
    # Verify user is member of team
    membership_result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == current_user.id,
        )
    )
    membership = membership_result.scalar_one_or_none()

    if membership is None:
        raise ForbiddenException("Not a member of this team")

    prefs = await get_or_create_preferences(db, current_user.id, team_id)

    return NotificationPreferencesResponse(
        email_enabled=prefs.email_enabled,
        deadline_reminder_hours=prefs.deadline_reminder_hours,
        no_log_days=prefs.no_log_days,
        weekly_digest_enabled=prefs.weekly_digest_enabled,
    )


@router.patch("", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    team_id: uuid.UUID,
    update_data: NotificationPreferencesUpdate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferencesResponse:
    """Update current user's notification preferences for a team.

    Users can only update their own preferences.
    Any fields not provided will remain unchanged.
    """
    # Verify user is member of team
    membership_result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == current_user.id,
        )
    )
    membership = membership_result.scalar_one_or_none()

    if membership is None:
        raise ForbiddenException("Not a member of this team")

    prefs = await get_or_create_preferences(db, current_user.id, team_id)

    # Update only provided fields
    if update_data.email_enabled is not None:
        prefs.email_enabled = update_data.email_enabled
    if update_data.deadline_reminder_hours is not None:
        prefs.deadline_reminder_hours = update_data.deadline_reminder_hours
    if update_data.no_log_days is not None:
        prefs.no_log_days = update_data.no_log_days
    if update_data.weekly_digest_enabled is not None:
        prefs.weekly_digest_enabled = update_data.weekly_digest_enabled

    await db.commit()
    await db.refresh(prefs)

    return NotificationPreferencesResponse(
        email_enabled=prefs.email_enabled,
        deadline_reminder_hours=prefs.deadline_reminder_hours,
        no_log_days=prefs.no_log_days,
        weekly_digest_enabled=prefs.weekly_digest_enabled,
    )

