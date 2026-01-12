"""Dashboard schemas for request/response models.

Implements contracts for:
- GET /teams/{team_id}/dashboards/member
- GET /teams/{team_id}/dashboards/leader

Privacy Rules (from systemprompt.md):
- private_ticket_aggregates must NEVER include ids or names
- Rows with count < 3 are omitted to prevent small-group identification
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from app.models.enums import (
    AssignmentScope,
    AssignmentType,
    Priority,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
)

# =============================================================================
# Due Bucket Enum
# =============================================================================


class DueBucket(str, Enum):
    """Due date bucket classification for aggregates."""

    OVERDUE = "overdue"  # due_at < now()
    DUE_TODAY = "due_today"  # due_at is today
    DUE_THIS_WEEK = "due_this_week"  # due_at within 7 days (excluding today)
    FUTURE = "future"  # due_at more than 7 days from now
    NO_DUE_DATE = "no_due_date"  # due_at IS NULL


# =============================================================================
# Shared Schemas
# =============================================================================


class CycleInfo(BaseModel):
    """Cycle information for dashboard responses."""

    id: uuid.UUID
    date: datetime
    label: str


# =============================================================================
# Member Dashboard Schemas
# =============================================================================


class AssignmentSummary(BaseModel):
    """Assignment summary for member dashboard."""

    id: uuid.UUID
    title: str
    priority: Priority
    type: AssignmentType
    scope: AssignmentScope
    section: str | None
    due_at: datetime | None


class TicketDueSoon(BaseModel):
    """Ticket due soon for member dashboard."""

    id: uuid.UUID
    title: str
    priority: Priority
    status: TicketStatus
    due_at: datetime | None
    visibility: TicketVisibility


class QuickLogDefaults(BaseModel):
    """Default values for quick practice logging."""

    duration_min_default: int = 20


class WeeklySummary(BaseModel):
    """Weekly practice summary for member dashboard."""

    practice_days: int  # Distinct days practiced in last 7 days
    streak_days: int  # Consecutive days practiced (ending today or yesterday)
    total_sessions: int  # Total practice sessions in last 7 days


class ProgressSummary(BaseModel):
    """Progress summary for member dashboard."""

    tickets_resolved_this_cycle: int
    tickets_verified_this_cycle: int


class MemberDashboardResponse(BaseModel):
    """Response schema for member dashboard.

    GET /teams/{team_id}/dashboards/member
    """

    cycle: CycleInfo | None
    countdown_days: int | None  # Days until cycle.date (negative if past)
    assignments: list[AssignmentSummary]
    tickets_due_soon: list[TicketDueSoon]
    quick_log_defaults: QuickLogDefaults
    weekly_summary: WeeklySummary
    progress: ProgressSummary


# =============================================================================
# Leader Dashboard Schemas
# =============================================================================


class MemberPracticeDays(BaseModel):
    """Practice days by member for leader dashboard."""

    member_id: uuid.UUID
    name: str
    section: str | None
    days_logged_7d: int


class ComplianceSummary(BaseModel):
    """Team compliance summary for leader dashboard."""

    logged_last_7_days_pct: float  # Percentage of members who logged at least 1 session
    practice_days_by_member: list[MemberPracticeDays]
    total_practice_minutes_7d: int  # Sum of duration_min in last 7 days


class SectionRisk(BaseModel):
    """Risk breakdown by section."""

    section: str
    blocking_due: int
    blocked: int
    resolved_not_verified: int


class SongRisk(BaseModel):
    """Risk breakdown by song."""

    song_ref: str
    blocking_due: int
    blocked: int
    resolved_not_verified: int


class RiskSummary(BaseModel):
    """Risk summary for leader dashboard."""

    blocking_due_count: int  # priority=BLOCKING and due_at within 7 days
    blocked_count: int  # status=BLOCKED
    resolved_not_verified_count: int  # status=RESOLVED
    by_section: list[SectionRisk]
    by_song: list[SongRisk]


class PrivateTicketAggregate(BaseModel):
    """Aggregated PRIVATE ticket data.

    PRIVACY RULE: This schema MUST NEVER include:
    - id (ticket ID)
    - owner_id
    - created_by
    - title
    - description
    - Any field that could identify the ticket owner

    Rows with count < 3 are omitted from the response.
    """

    section: str | None
    category: TicketCategory
    status: TicketStatus
    priority: Priority
    song_ref: str | None
    due_bucket: DueBucket
    count: int


class MemberDrilldown(BaseModel):
    """Member drilldown for leader dashboard."""

    member_id: uuid.UUID
    name: str
    section: str | None
    open_ticket_count: int
    blocked_count: int


class TicketVisible(BaseModel):
    """Visible ticket for leader drilldown (excludes PRIVATE)."""

    id: uuid.UUID
    title: str
    priority: Priority
    status: TicketStatus
    visibility: TicketVisibility
    section: str | None
    due_at: datetime | None


class DrilldownData(BaseModel):
    """Drilldown data for leader dashboard."""

    members: list[MemberDrilldown]
    tickets_visible: list[TicketVisible]


class LeaderDashboardResponse(BaseModel):
    """Response schema for leader dashboard.

    GET /teams/{team_id}/dashboards/leader

    Privacy Rules:
    - private_ticket_aggregates contains ONLY aggregate data
    - No identifiable information (no IDs, names, titles)
    - Rows with count < 3 are omitted
    """

    cycle: CycleInfo | None
    compliance: ComplianceSummary
    risk_summary: RiskSummary
    private_ticket_aggregates: list[PrivateTicketAggregate]
    drilldown: DrilldownData


# =============================================================================
# Compliance Insights Schemas
# =============================================================================


class ComplianceSectionDatum(BaseModel):
    """Aggregated practice compliance data by section."""

    section: str
    member_count: int
    total_practice_days_7d: int
    avg_practice_days_7d: float


class ComplianceInsightsResponse(BaseModel):
    """Response schema for compliance insights."""

    sections: list[ComplianceSectionDatum]
    summary: str
    summary_source: str
    window_days: int

