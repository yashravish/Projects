"""APScheduler setup for notification jobs.

This module configures the scheduler with all notification jobs and
provides the lifespan handler for FastAPI integration.

Jobs:
- no_log_reminder: Daily reminder for users who haven't logged practice
- blocking_due_48h: Every 6h alert for blocking tickets due soon
- blocked_over_48h: Every 6h alert for tickets blocked too long
- weekly_leader_digest: Weekly summary for section leaders

All jobs respect user notification preferences and RBAC boundaries.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import and_, func, select
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models import (
    NotificationPreference,
    PracticeLog,
    RehearsalCycle,
    Team,
    TeamMembership,
    Ticket,
    TicketActivity,
)
from app.models.enums import Priority, Role, TicketStatus
from app.services.email import EmailMessage, send_email

logger = logging.getLogger(__name__)

# Module-level scheduler instance
scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    """Get the scheduler instance, creating if needed."""
    global scheduler
    if scheduler is None:
        scheduler = BackgroundScheduler()
    return scheduler


class JobExecutionLog:
    """Helper for structured job logging."""

    def __init__(self, job_name: str):
        self.job_name = job_name
        self.start_time: float = 0
        self.status = "running"
        self.error: str | None = None

    def start(self) -> None:
        """Mark job as started."""
        self.start_time = time.time()
        logger.info(
            f"[Job:{self.job_name}] started",
            extra={
                "job_name": self.job_name,
                "last_run": datetime.now(UTC).isoformat(),
            },
        )

    def success(self) -> None:
        """Mark job as succeeded."""
        duration_ms = int((time.time() - self.start_time) * 1000)
        self.status = "success"
        next_run = self._get_next_run()
        logger.info(
            f"[Job:{self.job_name}] completed successfully",
            extra={
                "job_name": self.job_name,
                "duration_ms": duration_ms,
                "status": "success",
                "next_run": next_run,
            },
        )

    def failure(self, error: str) -> None:
        """Mark job as failed."""
        duration_ms = int((time.time() - self.start_time) * 1000)
        self.status = "failure"
        self.error = error
        next_run = self._get_next_run()
        logger.error(
            f"[Job:{self.job_name}] failed: {error}",
            extra={
                "job_name": self.job_name,
                "duration_ms": duration_ms,
                "status": "failure",
                "error": error,
                "next_run": next_run,
            },
        )

    def _get_next_run(self) -> str | None:
        """Get next scheduled run time for this job."""
        global scheduler
        if scheduler:
            job = scheduler.get_job(self.job_name)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        return None


def run_async_job(coro_func: Callable[[], Any]) -> Callable[[], None]:
    """Wrapper to run async jobs in the scheduler's thread."""

    def wrapper() -> None:
        asyncio.run(coro_func())

    return wrapper


# =============================================================================
# JOB 1: no_log_reminder - Daily reminder for users without practice logs
# =============================================================================


async def no_log_reminder_job() -> None:
    """Send reminders to users who haven't logged practice in N days.

    N is configured per-user via notification_preferences.no_log_days.
    Only sends if email_enabled = true for that user.
    """
    job_log = JobExecutionLog("no_log_reminder")
    job_log.start()

    try:
        async with async_session_maker() as db:
            # Get all notification preferences with email enabled
            prefs_result = await db.execute(
                select(NotificationPreference)
                .options(selectinload(NotificationPreference.user))
                .where(NotificationPreference.email_enabled == True)  # noqa: E712
            )
            preferences = prefs_result.scalars().all()

            now = datetime.now(UTC)
            emails_sent = 0

            for pref in preferences:
                # Check if user has logged practice in the last N days
                threshold_date = now - timedelta(days=pref.no_log_days)

                log_result = await db.execute(
                    select(func.count())
                    .select_from(PracticeLog)
                    .where(
                        and_(
                            PracticeLog.user_id == pref.user_id,
                            PracticeLog.team_id == pref.team_id,
                            PracticeLog.occurred_at >= threshold_date,
                        )
                    )
                )
                log_count = log_result.scalar() or 0

                if log_count == 0 and pref.user:
                    # Send reminder email
                    message = EmailMessage(
                        to=pref.user.email,
                        subject="🎵 Time to practice!",
                        body_text=(
                            f"Hi {pref.user.display_name},\n\n"
                            f"We noticed you haven't logged any practice in the last "
                            f"{pref.no_log_days} days.\n\n"
                            "Regular practice helps you improve and keeps your team informed "
                            "about your progress.\n\n"
                            "Log your practice now at PracticeOps!\n\n"
                            "Keep up the great work! 🎶"
                        ),
                    )
                    if send_email(message):
                        emails_sent += 1

            logger.info(f"[no_log_reminder] Sent {emails_sent} reminder emails")
            job_log.success()

    except Exception as e:
        job_log.failure(str(e))
        raise


# =============================================================================
# JOB 2: blocking_due_48h - Alert for blocking tickets due within 48 hours
# =============================================================================


async def blocking_due_48h_job() -> None:
    """Send alerts for blocking tickets due within 48 hours.

    Only tickets with:
    - priority = BLOCKING
    - due_at within next 48 hours
    - status != VERIFIED

    Email sent to ticket owner if they have email_enabled.
    """
    job_log = JobExecutionLog("blocking_due_48h")
    job_log.start()

    try:
        async with async_session_maker() as db:
            now = datetime.now(UTC)
            deadline = now + timedelta(hours=48)

            # Find blocking tickets due within 48h
            tickets_result = await db.execute(
                select(Ticket)
                .options(selectinload(Ticket.owner))
                .where(
                    and_(
                        Ticket.priority == Priority.BLOCKING,
                        Ticket.due_at.isnot(None),
                        Ticket.due_at <= deadline,
                        Ticket.status != TicketStatus.VERIFIED,
                        Ticket.owner_id.isnot(None),
                    )
                )
            )
            tickets = tickets_result.scalars().all()

            emails_sent = 0

            for ticket in tickets:
                if not ticket.owner:
                    continue

                # Check if owner has email enabled
                pref_result = await db.execute(
                    select(NotificationPreference).where(
                        and_(
                            NotificationPreference.user_id == ticket.owner_id,
                            NotificationPreference.team_id == ticket.team_id,
                            NotificationPreference.email_enabled == True,  # noqa: E712
                        )
                    )
                )
                pref = pref_result.scalar_one_or_none()

                if not pref:
                    continue

                # Calculate hours until due
                hours_until_due = (
                    (ticket.due_at - now).total_seconds() / 3600 if ticket.due_at else 0
                )

                message = EmailMessage(
                    to=ticket.owner.email,
                    subject=f"⚠️ BLOCKING ticket due soon: {ticket.title}",
                    body_text=(
                        f"Hi {ticket.owner.display_name},\n\n"
                        f"You have a BLOCKING priority ticket due in "
                        f"{int(hours_until_due)} hours:\n\n"
                        f"Title: {ticket.title}\n"
                        f"Status: {ticket.status.value}\n"
                        f"Due: {ticket.due_at.strftime('%Y-%m-%d %H:%M') if ticket.due_at else 'N/A'}\n\n"
                        "Please resolve this ticket as soon as possible to avoid "
                        "blocking your team's progress.\n\n"
                        "View in PracticeOps to take action."
                    ),
                )
                if send_email(message):
                    emails_sent += 1

            logger.info(f"[blocking_due_48h] Sent {emails_sent} alert emails")
            job_log.success()

    except Exception as e:
        job_log.failure(str(e))
        raise


# =============================================================================
# JOB 3: blocked_over_48h - Alert for tickets blocked for more than 48 hours
# =============================================================================


async def blocked_over_48h_job() -> None:
    """Send alerts for tickets that have been BLOCKED for over 48 hours.

    Uses TicketActivity to determine when ticket entered BLOCKED status.
    Email sent to section leader of the ticket's section.
    """
    job_log = JobExecutionLog("blocked_over_48h")
    job_log.start()

    try:
        async with async_session_maker() as db:
            now = datetime.now(UTC)
            threshold = now - timedelta(hours=48)

            # Find tickets currently in BLOCKED status
            blocked_tickets_result = await db.execute(
                select(Ticket).where(Ticket.status == TicketStatus.BLOCKED)
            )
            blocked_tickets = blocked_tickets_result.scalars().all()

            emails_sent = 0

            for ticket in blocked_tickets:
                # Find when this ticket became BLOCKED via TicketActivity
                activity_result = await db.execute(
                    select(TicketActivity)
                    .where(
                        and_(
                            TicketActivity.ticket_id == ticket.id,
                            TicketActivity.new_status == TicketStatus.BLOCKED,
                        )
                    )
                    .order_by(TicketActivity.created_at.desc())
                    .limit(1)
                )
                activity = activity_result.scalar_one_or_none()

                if not activity:
                    # No activity record found - skip
                    logger.warning(
                        f"[blocked_over_48h] No BLOCKED activity found for ticket {ticket.id}"
                    )
                    continue

                # Check if blocked for more than 48 hours
                if activity.created_at > threshold:
                    # Not blocked long enough yet
                    continue

                # Find section leader for this ticket's section
                if not ticket.section:
                    logger.warning(
                        f"[blocked_over_48h] Ticket {ticket.id} has no section, skipping"
                    )
                    continue

                leader_result = await db.execute(
                    select(TeamMembership)
                    .options(selectinload(TeamMembership.user))
                    .where(
                        and_(
                            TeamMembership.team_id == ticket.team_id,
                            TeamMembership.section == ticket.section,
                            TeamMembership.role == Role.SECTION_LEADER,
                        )
                    )
                )
                leader_membership = leader_result.scalar_one_or_none()

                if not leader_membership or not leader_membership.user:
                    logger.warning(
                        f"[blocked_over_48h] No section leader for section "
                        f"'{ticket.section}' in team {ticket.team_id}"
                    )
                    continue

                # Check if leader has email enabled
                pref_result = await db.execute(
                    select(NotificationPreference).where(
                        and_(
                            NotificationPreference.user_id == leader_membership.user_id,
                            NotificationPreference.team_id == ticket.team_id,
                            NotificationPreference.email_enabled == True,  # noqa: E712
                        )
                    )
                )
                pref = pref_result.scalar_one_or_none()

                if not pref:
                    continue

                hours_blocked = int((now - activity.created_at).total_seconds() / 3600)

                message = EmailMessage(
                    to=leader_membership.user.email,
                    subject=f"🚫 Ticket blocked for {hours_blocked}h: {ticket.title}",
                    body_text=(
                        f"Hi {leader_membership.user.display_name},\n\n"
                        f"A ticket in your section has been BLOCKED for over 48 hours:\n\n"
                        f"Title: {ticket.title}\n"
                        f"Section: {ticket.section}\n"
                        f"Blocked since: {activity.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                        f"Duration: {hours_blocked} hours\n\n"
                        "Please review and help unblock this ticket.\n\n"
                        "View in PracticeOps to take action."
                    ),
                )
                if send_email(message):
                    emails_sent += 1

            logger.info(f"[blocked_over_48h] Sent {emails_sent} alert emails")
            job_log.success()

    except Exception as e:
        job_log.failure(str(e))
        raise


# =============================================================================
# JOB 4: weekly_leader_digest - Weekly summary for section leaders
# =============================================================================


async def weekly_leader_digest_job() -> None:
    """Send weekly digest emails to section leaders.

    Only sends if:
    - weekly_digest_enabled = true
    - email_enabled = true
    - Team has an active cycle

    Includes compliance % and risk summary.
    """
    job_log = JobExecutionLog("weekly_leader_digest")
    job_log.start()

    try:
        async with async_session_maker() as db:
            now = datetime.now(UTC)
            week_ago = now - timedelta(days=7)

            # Find all teams with active cycles
            teams_result = await db.execute(select(Team))
            teams = teams_result.scalars().all()

            emails_sent = 0

            for team in teams:
                # Check for active cycle
                cycle_result = await db.execute(
                    select(RehearsalCycle)
                    .where(
                        and_(
                            RehearsalCycle.team_id == team.id,
                            RehearsalCycle.date <= now,
                        )
                    )
                    .order_by(RehearsalCycle.date.desc())
                    .limit(1)
                )
                active_cycle = cycle_result.scalar_one_or_none()

                if not active_cycle:
                    logger.info(
                        f"[weekly_leader_digest] Skipping team {team.id} - no active cycle"
                    )
                    continue

                # Get leaders with digest enabled
                leaders_result = await db.execute(
                    select(TeamMembership)
                    .options(selectinload(TeamMembership.user))
                    .where(
                        and_(
                            TeamMembership.team_id == team.id,
                            TeamMembership.role.in_([Role.SECTION_LEADER, Role.ADMIN]),
                        )
                    )
                )
                leaders = leaders_result.scalars().all()

                for leader in leaders:
                    if not leader.user:
                        continue

                    # Check notification preferences
                    pref_result = await db.execute(
                        select(NotificationPreference).where(
                            and_(
                                NotificationPreference.user_id == leader.user_id,
                                NotificationPreference.team_id == team.id,
                                NotificationPreference.email_enabled == True,  # noqa: E712
                                NotificationPreference.weekly_digest_enabled == True,  # noqa: E712
                            )
                        )
                    )
                    pref = pref_result.scalar_one_or_none()

                    if not pref:
                        continue

                    # Calculate compliance %
                    members_result = await db.execute(
                        select(func.count())
                        .select_from(TeamMembership)
                        .where(TeamMembership.team_id == team.id)
                    )
                    total_members = members_result.scalar() or 0

                    if total_members == 0:
                        continue

                    # Members who logged at least once in last 7 days
                    logged_members_result = await db.execute(
                        select(func.count(func.distinct(PracticeLog.user_id)))
                        .select_from(PracticeLog)
                        .where(
                            and_(
                                PracticeLog.team_id == team.id,
                                PracticeLog.occurred_at >= week_ago,
                            )
                        )
                    )
                    logged_members = logged_members_result.scalar() or 0
                    compliance_pct = round((logged_members / total_members) * 100, 1)

                    # Risk counts
                    blocking_result = await db.execute(
                        select(func.count())
                        .select_from(Ticket)
                        .where(
                            and_(
                                Ticket.team_id == team.id,
                                Ticket.cycle_id == active_cycle.id,
                                Ticket.priority == Priority.BLOCKING,
                                Ticket.status != TicketStatus.VERIFIED,
                            )
                        )
                    )
                    blocking_count = blocking_result.scalar() or 0

                    blocked_result = await db.execute(
                        select(func.count())
                        .select_from(Ticket)
                        .where(
                            and_(
                                Ticket.team_id == team.id,
                                Ticket.cycle_id == active_cycle.id,
                                Ticket.status == TicketStatus.BLOCKED,
                            )
                        )
                    )
                    blocked_count = blocked_result.scalar() or 0

                    unverified_result = await db.execute(
                        select(func.count())
                        .select_from(Ticket)
                        .where(
                            and_(
                                Ticket.team_id == team.id,
                                Ticket.cycle_id == active_cycle.id,
                                Ticket.status == TicketStatus.RESOLVED,
                            )
                        )
                    )
                    unverified_count = unverified_result.scalar() or 0

                    message = EmailMessage(
                        to=leader.user.email,
                        subject=f"📊 Weekly Digest - {team.name}",
                        body_text=(
                            f"Hi {leader.user.display_name},\n\n"
                            f"Here's your weekly summary for {team.name}:\n\n"
                            f"📈 COMPLIANCE\n"
                            f"  Practice logging: {compliance_pct}% "
                            f"({logged_members}/{total_members} members logged)\n\n"
                            f"⚠️ RISK SUMMARY\n"
                            f"  Blocking tickets: {blocking_count}\n"
                            f"  Blocked tickets: {blocked_count}\n"
                            f"  Resolved (not verified): {unverified_count}\n\n"
                            "Review your team dashboard in PracticeOps for details.\n\n"
                            "Have a great week! 🎵"
                        ),
                    )
                    if send_email(message):
                        emails_sent += 1

            logger.info(f"[weekly_leader_digest] Sent {emails_sent} digest emails")
            job_log.success()

    except Exception as e:
        job_log.failure(str(e))
        raise


# =============================================================================
# JOB REGISTRY - Maps job names to their async functions
# =============================================================================

JOB_REGISTRY: dict[str, Callable[[], Any]] = {
    "no_log_reminder": no_log_reminder_job,
    "blocking_due_48h": blocking_due_48h_job,
    "blocked_over_48h": blocked_over_48h_job,
    "weekly_leader_digest": weekly_leader_digest_job,
}


async def run_job_by_name(job_name: str) -> dict[str, Any]:
    """Run a job by name (for manual trigger endpoint).

    Args:
        job_name: Name of the job to run

    Returns:
        Dict with status and optional error message

    Raises:
        ValueError: If job_name is not found in registry
    """
    if job_name not in JOB_REGISTRY:
        raise ValueError(f"Unknown job: {job_name}")

    job_func = JOB_REGISTRY[job_name]
    start_time = time.time()

    try:
        await job_func()
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "job_name": job_name,
            "status": "success",
            "duration_ms": duration_ms,
        }
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "job_name": job_name,
            "status": "failure",
            "duration_ms": duration_ms,
            "error": str(e),
        }


# =============================================================================
# SCHEDULER SETUP
# =============================================================================


def setup_scheduler() -> BackgroundScheduler:
    """Configure and return the scheduler with all jobs.

    Jobs are registered exactly once to prevent duplicates.
    """
    sched = get_scheduler()

    # Only add jobs if they don't already exist
    existing_jobs = {job.id for job in sched.get_jobs()}

    # Job 1: no_log_reminder - Daily at 9 AM UTC
    if "no_log_reminder" not in existing_jobs:
        sched.add_job(
            run_async_job(no_log_reminder_job),
            trigger=CronTrigger(hour=9, minute=0),
            id="no_log_reminder",
            name="Daily practice reminder",
            replace_existing=True,
        )
        logger.info("Registered job: no_log_reminder (daily at 09:00 UTC)")

    # Job 2: blocking_due_48h - Every 6 hours
    if "blocking_due_48h" not in existing_jobs:
        sched.add_job(
            run_async_job(blocking_due_48h_job),
            trigger=IntervalTrigger(hours=6),
            id="blocking_due_48h",
            name="Blocking ticket due soon alert",
            replace_existing=True,
        )
        logger.info("Registered job: blocking_due_48h (every 6 hours)")

    # Job 3: blocked_over_48h - Every 6 hours
    if "blocked_over_48h" not in existing_jobs:
        sched.add_job(
            run_async_job(blocked_over_48h_job),
            trigger=IntervalTrigger(hours=6),
            id="blocked_over_48h",
            name="Blocked ticket alert",
            replace_existing=True,
        )
        logger.info("Registered job: blocked_over_48h (every 6 hours)")

    # Job 4: weekly_leader_digest - Sundays at 8 AM UTC
    if "weekly_leader_digest" not in existing_jobs:
        sched.add_job(
            run_async_job(weekly_leader_digest_job),
            trigger=CronTrigger(day_of_week="sun", hour=8, minute=0),
            id="weekly_leader_digest",
            name="Weekly leader digest",
            replace_existing=True,
        )
        logger.info("Registered job: weekly_leader_digest (Sundays at 08:00 UTC)")

    return sched


@asynccontextmanager
async def scheduler_lifespan(app: Any):  # noqa: ARG001
    """FastAPI lifespan handler for scheduler startup/shutdown.

    Usage:
        app = FastAPI(lifespan=scheduler_lifespan)
    """
    # Startup
    logger.info("Starting notification scheduler...")
    sched = setup_scheduler()
    sched.start()
    logger.info(f"Scheduler started with {len(sched.get_jobs())} jobs")

    yield

    # Shutdown
    logger.info("Shutting down notification scheduler...")
    sched.shutdown(wait=True)
    logger.info("Scheduler shutdown complete")

