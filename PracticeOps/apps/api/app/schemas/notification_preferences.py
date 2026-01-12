"""Notification preferences schemas."""

from pydantic import BaseModel, Field


class NotificationPreferencesResponse(BaseModel):
    """Response schema for notification preferences."""

    email_enabled: bool = Field(..., description="Whether email notifications are enabled")
    deadline_reminder_hours: int = Field(
        ..., description="Hours before deadline to send reminder"
    )
    no_log_days: int = Field(
        ..., description="Days without practice before sending reminder"
    )
    weekly_digest_enabled: bool = Field(
        ..., description="Whether to receive weekly leader digest emails"
    )

    model_config = {"from_attributes": True}


class NotificationPreferencesUpdate(BaseModel):
    """Request schema for updating notification preferences."""

    email_enabled: bool | None = Field(
        None, description="Whether email notifications are enabled"
    )
    deadline_reminder_hours: int | None = Field(
        None, ge=1, le=168, description="Hours before deadline (1-168)"
    )
    no_log_days: int | None = Field(
        None, ge=1, le=30, description="Days without practice (1-30)"
    )
    weekly_digest_enabled: bool | None = Field(
        None, description="Whether to receive weekly digest emails"
    )

