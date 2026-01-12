"""Dashboard routes.

GET /teams/{team_id}/dashboards/member - Member dashboard
GET /teams/{team_id}/dashboards/leader - Leader dashboard (ADMIN/SECTION_LEADER)

RBAC is enforced via dependencies.
Privacy rules strictly enforced for private_ticket_aggregates.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import CurrentUser
from app.core.errors import ForbiddenException, NotFoundException
from app.database import get_db
from app.models import (
    Assignment,
    AssignmentScope,
    PracticeLog,
    Priority,
    RehearsalCycle,
    Role,
    Team,
    TeamMembership,
    Ticket,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
    User,
)
from app.schemas.dashboards import (
    AssignmentSummary,
    ComplianceSummary,
    ComplianceInsightsResponse,
    ComplianceSectionDatum,
    CycleInfo,
    DrilldownData,
    DueBucket,
    LeaderDashboardResponse,
    MemberDashboardResponse,
    MemberDrilldown,
    MemberPracticeDays,
    PrivateTicketAggregate,
    ProgressSummary,
    QuickLogDefaults,
    RiskSummary,
    SectionRisk,
    SongRisk,
    TicketDueSoon,
    TicketVisible,
    WeeklySummary,
)
from app.services.openai_summary import generate_compliance_summary

router = APIRouter(prefix="/teams/{team_id}/dashboards", tags=["dashboards"])

# Minimum count threshold for private ticket aggregates (prevent small-group identification)
PRIVACY_MIN_COUNT_THRESHOLD = 3


# =============================================================================
# Helper functions
# =============================================================================


async def _get_team(team_id: uuid.UUID, db: AsyncSession) -> Team | None:
    """Get team by ID."""
    result = await db.execute(select(Team).where(Team.id == team_id))
    return result.scalar_one_or_none()


async def _get_membership(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership | None:
    """Get user's membership for a team."""
    result = await db.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _require_membership(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> TeamMembership:
    """Require user to be a member of the team."""
    membership = await _get_membership(team_id, user_id, db)
    if membership is None:
        raise ForbiddenException("Not a member of this team")
    return membership


async def _get_active_cycle(team_id: uuid.UUID, db: AsyncSession) -> RehearsalCycle | None:
    """Get active cycle for a team.

    Active cycle logic (from Milestone 4):
    1. Nearest upcoming cycle (date >= today)
    2. Else latest past cycle (date < today)
    3. Else null
    """
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Try upcoming first
    result = await db.execute(
        select(RehearsalCycle)
        .where(RehearsalCycle.team_id == team_id, RehearsalCycle.date >= today)
        .order_by(RehearsalCycle.date.asc(), RehearsalCycle.id.asc())
        .limit(1)
    )
    cycle = result.scalar_one_or_none()

    if cycle is None:
        # Try latest past
        result = await db.execute(
            select(RehearsalCycle)
            .where(RehearsalCycle.team_id == team_id, RehearsalCycle.date < today)
            .order_by(RehearsalCycle.date.desc(), RehearsalCycle.id.desc())
            .limit(1)
        )
        cycle = result.scalar_one_or_none()

    return cycle


async def _get_cycle_by_id(cycle_id: uuid.UUID, db: AsyncSession) -> RehearsalCycle | None:
    """Get cycle by ID."""
    result = await db.execute(select(RehearsalCycle).where(RehearsalCycle.id == cycle_id))
    return result.scalar_one_or_none()


def _calculate_countdown_days(cycle_date: datetime) -> int:
    """Calculate days until cycle date.

    Returns:
    - Positive: days in the future
    - 0: today
    - Negative: days in the past
    """
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    cycle_day = cycle_date.replace(hour=0, minute=0, second=0, microsecond=0)
    delta = cycle_day - today
    return delta.days


def _classify_due_bucket(due_at: datetime | None, now: datetime) -> DueBucket:
    """Classify a due date into a bucket."""
    if due_at is None:
        return DueBucket.NO_DUE_DATE

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    week_end = today_start + timedelta(days=7)

    if due_at < now:
        return DueBucket.OVERDUE
    elif due_at < today_end:
        return DueBucket.DUE_TODAY
    elif due_at < week_end:
        return DueBucket.DUE_THIS_WEEK
    else:
        return DueBucket.FUTURE


# =============================================================================
# GET /teams/{team_id}/dashboards/member
# =============================================================================


@router.get("/member", response_model=MemberDashboardResponse)
async def get_member_dashboard(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    cycle_id: Annotated[uuid.UUID | None, Query()] = None,
) -> MemberDashboardResponse:
    """Get member dashboard.

    Shows personal progress, assignments, and tickets due.
    """
    # Verify team exists
    team = await _get_team(team_id, db)
    if team is None:
        raise NotFoundException("Team not found")

    # Require membership
    membership = await _require_membership(team_id, current_user.id, db)

    # Get cycle (specified or active)
    if cycle_id:
        cycle = await _get_cycle_by_id(cycle_id, db)
        if cycle is None or cycle.team_id != team_id:
            raise NotFoundException("Cycle not found")
    else:
        cycle = await _get_active_cycle(team_id, db)

    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)

    # Build cycle info
    cycle_info: CycleInfo | None = None
    countdown_days: int | None = None
    if cycle:
        cycle_info = CycleInfo(id=cycle.id, date=cycle.date, label=cycle.name)
        countdown_days = _calculate_countdown_days(cycle.date)

    # Get assignments visible to user (TEAM + user's SECTION)
    assignments_data: list[AssignmentSummary] = []
    if cycle:
        assignment_query = select(Assignment).where(Assignment.cycle_id == cycle.id)

        # Visibility: TEAM always, SECTION only if section matches
        visibility_filter = or_(
            Assignment.scope == AssignmentScope.TEAM,
            and_(
                Assignment.scope == AssignmentScope.SECTION,
                Assignment.section == membership.section,
            ),
        )
        assignment_query = assignment_query.where(visibility_filter)
        assignment_query = assignment_query.order_by(
            Assignment.priority.desc(), Assignment.due_at.asc().nulls_last()
        )

        result = await db.execute(assignment_query)
        assignments = result.scalars().all()

        for a in assignments:
            assignments_data.append(
                AssignmentSummary(
                    id=a.id,
                    title=a.title,
                    priority=a.priority,
                    type=a.type,
                    scope=a.scope,
                    section=a.section,
                    due_at=a.due_at,
                )
            )

    # Get tickets due soon (owned or created by user, due within 7 days)
    tickets_due_soon: list[TicketDueSoon] = []
    week_from_now = now + timedelta(days=7)

    ticket_query = (
        select(Ticket)
        .where(
            Ticket.team_id == team_id,
            or_(Ticket.owner_id == current_user.id, Ticket.created_by == current_user.id),
            Ticket.status.not_in([TicketStatus.VERIFIED, TicketStatus.RESOLVED]),
            Ticket.due_at.isnot(None),
            Ticket.due_at <= week_from_now,
        )
        .order_by(Ticket.due_at.asc(), Ticket.priority.desc())
        .limit(10)
    )
    tickets_result = await db.execute(ticket_query)
    tickets_list: list[Ticket] = list(tickets_result.scalars().all())

    for t in tickets_list:
        tickets_due_soon.append(
            TicketDueSoon(
                id=t.id,
                title=t.title,
                priority=t.priority,
                status=t.status,
                due_at=t.due_at,
                visibility=t.visibility,
            )
        )

    # Weekly summary - practice logs in last 7 days
    practice_logs_query = (
        select(PracticeLog)
        .where(
            PracticeLog.user_id == current_user.id,
            PracticeLog.team_id == team_id,
            PracticeLog.occurred_at >= seven_days_ago,
        )
        .order_by(PracticeLog.occurred_at.desc())
    )
    logs_result = await db.execute(practice_logs_query)
    practice_logs: list[PracticeLog] = list(logs_result.scalars().all())

    # Calculate practice days (distinct days)
    practice_days_set = {log.occurred_at.date() for log in practice_logs}
    practice_days = len(practice_days_set)
    total_sessions = len(practice_logs)

    # Calculate streak days (consecutive days ending today or yesterday)
    streak_days = 0
    today_date = now.date()
    check_date: date | None = today_date

    # Check if practiced today
    if today_date in practice_days_set:
        streak_days = 1
        check_date = today_date - timedelta(days=1)
    else:
        # Check if practiced yesterday
        yesterday = today_date - timedelta(days=1)
        if yesterday in practice_days_set:
            streak_days = 1
            check_date = yesterday - timedelta(days=1)
        else:
            check_date = None  # No recent practice, streak is 0

    # Count consecutive days
    while check_date is not None and check_date in practice_days_set:
        streak_days += 1
        check_date = check_date - timedelta(days=1)

    weekly_summary = WeeklySummary(
        practice_days=practice_days,
        streak_days=streak_days,
        total_sessions=total_sessions,
    )

    # Progress - tickets resolved/verified this cycle
    tickets_resolved = 0
    tickets_verified = 0
    if cycle:
        # Resolved this cycle (owned by user)
        resolved_query = select(func.count()).where(
            Ticket.owner_id == current_user.id,
            Ticket.cycle_id == cycle.id,
            Ticket.status.in_([TicketStatus.RESOLVED, TicketStatus.VERIFIED]),
        )
        resolved_result = await db.execute(resolved_query)
        resolved_count = resolved_result.scalar()
        tickets_resolved = int(resolved_count) if resolved_count else 0

        # Verified this cycle (owned by user)
        verified_query = select(func.count()).where(
            Ticket.owner_id == current_user.id,
            Ticket.cycle_id == cycle.id,
            Ticket.status == TicketStatus.VERIFIED,
        )
        verified_result = await db.execute(verified_query)
        verified_count = verified_result.scalar()
        tickets_verified = int(verified_count) if verified_count else 0

    progress = ProgressSummary(
        tickets_resolved_this_cycle=tickets_resolved,
        tickets_verified_this_cycle=tickets_verified,
    )

    return MemberDashboardResponse(
        cycle=cycle_info,
        countdown_days=countdown_days,
        assignments=assignments_data,
        tickets_due_soon=tickets_due_soon,
        quick_log_defaults=QuickLogDefaults(),
        weekly_summary=weekly_summary,
        progress=progress,
    )


# =============================================================================
# GET /teams/{team_id}/dashboards/leader
# =============================================================================


@router.get("/leader", response_model=LeaderDashboardResponse)
async def get_leader_dashboard(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    cycle_id: Annotated[uuid.UUID | None, Query()] = None,
) -> LeaderDashboardResponse:
    """Get leader dashboard.

    Shows team compliance, risk summary, and private ticket aggregates.
    Only accessible by ADMIN or SECTION_LEADER.
    """
    # Verify team exists
    team = await _get_team(team_id, db)
    if team is None:
        raise NotFoundException("Team not found")

    # Require leader role
    membership = await _require_membership(team_id, current_user.id, db)
    if membership.role not in [Role.ADMIN, Role.SECTION_LEADER]:
        raise ForbiddenException("Only leaders can access this dashboard")

    is_admin = membership.role == Role.ADMIN
    leader_section = membership.section

    # Get cycle (specified or active)
    if cycle_id:
        cycle = await _get_cycle_by_id(cycle_id, db)
        if cycle is None or cycle.team_id != team_id:
            raise NotFoundException("Cycle not found")
    else:
        cycle = await _get_active_cycle(team_id, db)

    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)
    week_from_now = now + timedelta(days=7)

    # Build cycle info
    cycle_info: CycleInfo | None = None
    if cycle:
        cycle_info = CycleInfo(id=cycle.id, date=cycle.date, label=cycle.name)

    # ==========================================================================
    # Compliance Summary
    # ==========================================================================

    # Get team members (scoped for section leaders)
    members_query = (
        select(TeamMembership, User)
        .join(User, TeamMembership.user_id == User.id)
        .where(TeamMembership.team_id == team_id)
    )
    if not is_admin:
        members_query = members_query.where(TeamMembership.section == leader_section)

    result = await db.execute(members_query)
    member_rows = result.all()

    # Practice days per member in last 7 days
    # OPTIMIZATION: Batch fetch all practice logs to avoid N+1 queries
    member_ids = [user_row.id for _, user_row in member_rows]

    # Single query for all member practice logs
    all_logs_query = select(PracticeLog).where(
        PracticeLog.user_id.in_(member_ids),
        PracticeLog.team_id == team_id,
        PracticeLog.occurred_at >= seven_days_ago,
    )
    all_logs_result = await db.execute(all_logs_query)
    all_logs = list(all_logs_result.scalars().all())

    # Group logs by user_id
    logs_by_user: dict[uuid.UUID, list[PracticeLog]] = {}
    for log in all_logs:
        if log.user_id not in logs_by_user:
            logs_by_user[log.user_id] = []
        logs_by_user[log.user_id].append(log)

    practice_days_by_member: list[MemberPracticeDays] = []
    members_with_logs = 0
    total_minutes = 0

    for membership_row, user_row in member_rows:
        # Get practice logs from pre-fetched data
        logs = logs_by_user.get(user_row.id, [])

        days_logged = len({log.occurred_at.date() for log in logs})
        member_minutes = sum(log.duration_minutes for log in logs)
        total_minutes += member_minutes

        if days_logged > 0:
            members_with_logs += 1

        practice_days_by_member.append(
            MemberPracticeDays(
                member_id=user_row.id,
                name=user_row.display_name,
                section=membership_row.section,
                days_logged_7d=days_logged,
            )
        )

    # Calculate percentage
    total_members = len(member_rows)
    logged_pct = members_with_logs / total_members if total_members > 0 else 0.0

    compliance = ComplianceSummary(
        logged_last_7_days_pct=round(logged_pct, 2),
        practice_days_by_member=practice_days_by_member,
        total_practice_minutes_7d=total_minutes,
    )

    # ==========================================================================
    # Risk Summary
    # ==========================================================================

    # Base ticket filter for visible tickets (non-PRIVATE or SECTION leader can see)
    visible_ticket_filter = or_(
        Ticket.visibility == TicketVisibility.TEAM,
        Ticket.visibility == TicketVisibility.SECTION,
    )
    if not is_admin:
        # Section leaders can only see TEAM + their SECTION
        visible_ticket_filter = or_(
            Ticket.visibility == TicketVisibility.TEAM,
            and_(
                Ticket.visibility == TicketVisibility.SECTION,
                Ticket.section == leader_section,
            ),
        )

    # Blocking due count (priority=BLOCKING, due within 7 days)
    blocking_due_query = select(func.count()).where(
        Ticket.team_id == team_id,
        Ticket.priority == Priority.BLOCKING,
        Ticket.due_at.isnot(None),
        Ticket.due_at <= week_from_now,
        Ticket.status.not_in([TicketStatus.VERIFIED]),
        visible_ticket_filter,
    )
    result = await db.execute(blocking_due_query)
    blocking_due_count = result.scalar() or 0

    # Blocked count
    blocked_query = select(func.count()).where(
        Ticket.team_id == team_id,
        Ticket.status == TicketStatus.BLOCKED,
        visible_ticket_filter,
    )
    result = await db.execute(blocked_query)
    blocked_count = result.scalar() or 0

    # Resolved not verified count
    resolved_query = select(func.count()).where(
        Ticket.team_id == team_id,
        Ticket.status == TicketStatus.RESOLVED,
        visible_ticket_filter,
    )
    result = await db.execute(resolved_query)
    resolved_not_verified_count = result.scalar() or 0

    # By section breakdown
    by_section: list[SectionRisk] = []

    # Get distinct sections for the team
    sections_query = (
        select(TeamMembership.section)
        .where(TeamMembership.team_id == team_id, TeamMembership.section.isnot(None))
        .distinct()
    )
    if not is_admin:
        sections_query = sections_query.where(TeamMembership.section == leader_section)

    result = await db.execute(sections_query)
    sections = [row[0] for row in result.all()]

    for section in sections:
        # Blocking due for this section
        section_blocking_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.section == section,
            Ticket.priority == Priority.BLOCKING,
            Ticket.due_at.isnot(None),
            Ticket.due_at <= week_from_now,
            Ticket.status.not_in([TicketStatus.VERIFIED]),
            visible_ticket_filter,
        )
        result = await db.execute(section_blocking_query)
        section_blocking = result.scalar() or 0

        # Blocked for this section
        section_blocked_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.section == section,
            Ticket.status == TicketStatus.BLOCKED,
            visible_ticket_filter,
        )
        result = await db.execute(section_blocked_query)
        section_blocked = result.scalar() or 0

        # Resolved not verified for this section
        section_resolved_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.section == section,
            Ticket.status == TicketStatus.RESOLVED,
            visible_ticket_filter,
        )
        result = await db.execute(section_resolved_query)
        section_resolved = result.scalar() or 0

        by_section.append(
            SectionRisk(
                section=section,
                blocking_due=section_blocking,
                blocked=section_blocked,
                resolved_not_verified=section_resolved,
            )
        )

    # By song breakdown
    by_song: list[SongRisk] = []

    # Get distinct song_refs
    songs_query = (
        select(Ticket.song_ref)
        .where(
            Ticket.team_id == team_id,
            Ticket.song_ref.isnot(None),
            visible_ticket_filter,
        )
        .distinct()
    )
    result = await db.execute(songs_query)
    song_refs = [row[0] for row in result.all()]

    for song_ref in song_refs:
        # Blocking due for this song
        song_blocking_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.song_ref == song_ref,
            Ticket.priority == Priority.BLOCKING,
            Ticket.due_at.isnot(None),
            Ticket.due_at <= week_from_now,
            Ticket.status.not_in([TicketStatus.VERIFIED]),
            visible_ticket_filter,
        )
        result = await db.execute(song_blocking_query)
        song_blocking = result.scalar() or 0

        # Blocked for this song
        song_blocked_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.song_ref == song_ref,
            Ticket.status == TicketStatus.BLOCKED,
            visible_ticket_filter,
        )
        result = await db.execute(song_blocked_query)
        song_blocked = result.scalar() or 0

        # Resolved not verified for this song
        song_resolved_query = select(func.count()).where(
            Ticket.team_id == team_id,
            Ticket.song_ref == song_ref,
            Ticket.status == TicketStatus.RESOLVED,
            visible_ticket_filter,
        )
        result = await db.execute(song_resolved_query)
        song_resolved = result.scalar() or 0

        by_song.append(
            SongRisk(
                song_ref=song_ref,
                blocking_due=song_blocking,
                blocked=song_blocked,
                resolved_not_verified=song_resolved,
            )
        )

    risk_summary = RiskSummary(
        blocking_due_count=blocking_due_count,
        blocked_count=blocked_count,
        resolved_not_verified_count=resolved_not_verified_count,
        by_section=by_section,
        by_song=by_song,
    )

    # ==========================================================================
    # Private Ticket Aggregates (PRIVACY-SENSITIVE)
    # ==========================================================================

    # Aggregate PRIVATE tickets by: section, category, status, priority, song_ref, due_bucket
    # MUST NOT include any identifying information!

    private_tickets_query = select(Ticket).where(
        Ticket.team_id == team_id,
        Ticket.visibility == TicketVisibility.PRIVATE,
    )
    result = await db.execute(private_tickets_query)
    private_tickets = list(result.scalars().all())

    # Group tickets by aggregate key
    AggregateKey = tuple[str | None, TicketCategory, TicketStatus, Priority, str | None, DueBucket]
    aggregates: dict[AggregateKey, int] = {}
    for ticket in private_tickets:
        due_bucket = _classify_due_bucket(ticket.due_at, now)
        key = (
            ticket.section,
            ticket.category,
            ticket.status,
            ticket.priority,
            ticket.song_ref,
            due_bucket,
        )
        aggregates[key] = aggregates.get(key, 0) + 1

    # Build response, filtering out rows with count < PRIVACY_MIN_COUNT_THRESHOLD
    private_ticket_aggregates: list[PrivateTicketAggregate] = []
    for (section, category, status, priority, song_ref, due_bucket), count in aggregates.items():
        if count >= PRIVACY_MIN_COUNT_THRESHOLD:
            private_ticket_aggregates.append(
                PrivateTicketAggregate(
                    section=section,
                    category=category,
                    status=status,
                    priority=priority,
                    song_ref=song_ref,
                    due_bucket=due_bucket,
                    count=count,
                )
            )

    # ==========================================================================
    # Drilldown Data
    # ==========================================================================

    # Members drilldown (scoped for section leaders)
    # OPTIMIZATION: Batch fetch ticket counts to avoid N+1 queries

    # Single query to get all ticket counts per member
    ticket_counts_query = (
        select(
            Ticket.owner_id,
            Ticket.status,
            func.count().label("count"),
        )
        .where(
            Ticket.owner_id.in_(member_ids),
            Ticket.team_id == team_id,
            visible_ticket_filter,
        )
        .group_by(Ticket.owner_id, Ticket.status)
    )
    ticket_counts_result = await db.execute(ticket_counts_query)
    ticket_counts_rows = ticket_counts_result.all()

    # Build lookup: {user_id: {status: count}}
    ticket_counts_by_user: dict[uuid.UUID, dict[TicketStatus, int]] = {}
    for owner_id, status, count in ticket_counts_rows:
        if owner_id not in ticket_counts_by_user:
            ticket_counts_by_user[owner_id] = {}
        ticket_counts_by_user[owner_id][status] = count

    members_drilldown: list[MemberDrilldown] = []

    for membership_row, user_row in member_rows:
        # Get counts from pre-fetched data
        user_counts = ticket_counts_by_user.get(user_row.id, {})

        # Open = OPEN + IN_PROGRESS
        open_count = user_counts.get(TicketStatus.OPEN, 0) + user_counts.get(
            TicketStatus.IN_PROGRESS, 0
        )

        # Blocked count
        member_blocked = user_counts.get(TicketStatus.BLOCKED, 0)

        members_drilldown.append(
            MemberDrilldown(
                member_id=user_row.id,
                name=user_row.display_name,
                section=membership_row.section,
                open_ticket_count=open_count,
                blocked_count=member_blocked,
            )
        )

    # Visible tickets (excludes PRIVATE)
    tickets_visible_query = (
        select(Ticket)
        .where(
            Ticket.team_id == team_id,
            Ticket.visibility != TicketVisibility.PRIVATE,
            visible_ticket_filter,
        )
        .order_by(Ticket.priority.desc(), Ticket.due_at.asc().nulls_last())
        .limit(50)
    )
    if not is_admin:
        # Section leaders only see TEAM + their SECTION tickets
        tickets_visible_query = (
            select(Ticket)
            .where(
                Ticket.team_id == team_id,
                or_(
                    Ticket.visibility == TicketVisibility.TEAM,
                    and_(
                        Ticket.visibility == TicketVisibility.SECTION,
                        Ticket.section == leader_section,
                    ),
                ),
            )
            .order_by(Ticket.priority.desc(), Ticket.due_at.asc().nulls_last())
            .limit(50)
        )

    result = await db.execute(tickets_visible_query)
    visible_tickets = result.scalars().all()

    tickets_visible_list: list[TicketVisible] = []
    for t in visible_tickets:
        tickets_visible_list.append(
            TicketVisible(
                id=t.id,
                title=t.title,
                priority=t.priority,
                status=t.status,
                visibility=t.visibility,
                section=t.section,
                due_at=t.due_at,
            )
        )

    drilldown = DrilldownData(
        members=members_drilldown,
        tickets_visible=tickets_visible_list,
    )

    return LeaderDashboardResponse(
        cycle=cycle_info,
        compliance=compliance,
        risk_summary=risk_summary,
        private_ticket_aggregates=private_ticket_aggregates,
        drilldown=drilldown,
    )


# =============================================================================
# GET /teams/{team_id}/dashboards/leader/compliance-insights
# =============================================================================


@router.get("/leader/compliance-insights", response_model=ComplianceInsightsResponse)
async def get_leader_compliance_insights(
    team_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ComplianceInsightsResponse:
    """Get compliance insights for leaders.

    Returns section-level compliance aggregates and an AI summary.
    """
    team = await _get_team(team_id, db)
    if team is None:
        raise NotFoundException("Team not found")

    membership = await _require_membership(team_id, current_user.id, db)
    if membership.role not in [Role.ADMIN, Role.SECTION_LEADER]:
        raise ForbiddenException("Only leaders can access compliance insights")

    is_admin = membership.role == Role.ADMIN
    leader_section = membership.section

    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)

    members_query = (
        select(TeamMembership, User)
        .join(User, TeamMembership.user_id == User.id)
        .where(TeamMembership.team_id == team_id)
    )
    if not is_admin:
        members_query = members_query.where(TeamMembership.section == leader_section)

    result = await db.execute(members_query)
    member_rows = result.all()

    member_ids = [user_row.id for _, user_row in member_rows]
    if not member_ids:
        return ComplianceInsightsResponse(
            sections=[],
            summary="No compliance data available for the last 7 days.",
            summary_source="fallback",
            window_days=7,
        )

    logs_query = select(PracticeLog).where(
        PracticeLog.user_id.in_(member_ids),
        PracticeLog.team_id == team_id,
        PracticeLog.occurred_at >= seven_days_ago,
    )
    logs_result = await db.execute(logs_query)
    logs = list(logs_result.scalars().all())

    logs_by_user: dict[uuid.UUID, list[PracticeLog]] = {}
    for log in logs:
        logs_by_user.setdefault(log.user_id, []).append(log)

    section_totals: dict[str, dict[str, int]] = {}
    for membership_row, user_row in member_rows:
        section_label = membership_row.section or "Unassigned"
        user_logs = logs_by_user.get(user_row.id, [])
        days_logged = len({log.occurred_at.date() for log in user_logs})

        if section_label not in section_totals:
            section_totals[section_label] = {"member_count": 0, "total_days": 0}

        section_totals[section_label]["member_count"] += 1
        section_totals[section_label]["total_days"] += days_logged

    sections: list[ComplianceSectionDatum] = []
    for section, totals in section_totals.items():
        member_count = totals["member_count"]
        total_days = totals["total_days"]
        avg_days = round(total_days / member_count, 2) if member_count else 0.0
        sections.append(
            ComplianceSectionDatum(
                section=section,
                member_count=member_count,
                total_practice_days_7d=total_days,
                avg_practice_days_7d=avg_days,
            )
        )

    summary_input = [
        {
            "section": s.section,
            "member_count": s.member_count,
            "total_practice_days_7d": s.total_practice_days_7d,
            "avg_practice_days_7d": s.avg_practice_days_7d,
        }
        for s in sections
    ]
    summary, summary_source = await generate_compliance_summary(summary_input)

    return ComplianceInsightsResponse(
        sections=sections,
        summary=summary,
        summary_source=summary_source.value,
        window_days=7,
    )

