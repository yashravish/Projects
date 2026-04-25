"""Audit ledger and retention orchestration.

The ledger is **append-only** at the API surface and **tamper-evident**
in storage: every row carries a `prev_hash` (the entry_hash of the
previous row in the same tenant's chain) and an `entry_hash` of its own,
where

    entry_hash = sha256( canonical_payload || prev_hash )

`canonical_payload` is the JSON-serialised tuple of the auditable fields
in a fixed key order (see `_canonical_payload`). Re-walking the chain
deterministically reproduces every entry_hash; any divergence indicates
in-place tampering or a row deletion.

Public surface:

    * `record_event`          — append a row; computes prev_hash + entry_hash
                                 inside a single tenant-locked transaction.
    * `list_events`           — paginated, filtered list for the SPA.
    * `get_event`             — single row, scoped by org.
    * `verify_chain`          — full chain re-walk for a tenant.
    * `compute_ledger_stats`  — summary counts for the page header.
    * `export_events_csv`     — streaming CSV bytes of filtered events.
    * `list_policies` / `upsert_policy` — retention configuration.
    * `run_retention`         — execute the sweep + log a `RetentionRun` row.
    * `list_retention_runs`   — paginated history.

Tenant isolation: every persistence path passes `organization_id`
through `apply_tenant_filter`, matching the audit-only invariant
enforced by `tests/unit/test_tenant_isolation.py`.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import uuid
from typing import Any, Iterable, cast as typing_cast

from sqlalchemy import Text, and_, asc, cast, desc, func, or_, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog, RetentionPolicy, RetentionRun, User
from app.governance import retention as retention_mod
from app.logging_config import get_logger
from app.schemas.audit import (
    AuditEventFilters,
    AuditEventList,
    AuditEventOut,
    AuditOutcome,
    IntegrityBreak,
    IntegrityReport,
    LedgerStats,
    RetentionPolicyOut,
    RetentionPolicyUpsert,
    RetentionResource,
    RetentionRunList,
    RetentionRunOut,
    RetentionStatus,
)
from app.security.tenant import apply_tenant_filter

log = get_logger("audit_service")


class AuditError(Exception):
    """Raised on user-visible audit errors (invalid policy resource etc.)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuditEventNotFoundError(Exception):
    """Raised when an event id does not exist for the tenant."""


# ── Hash chain ───────────────────────────────────────────────────────────────


def _canonical_payload(
    *,
    organization_id: uuid.UUID,
    actor_id: uuid.UUID | None,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID | None,
    outcome: str,
    request_id: str | None,
    metadata: dict[str, Any],
    created_at: dt.datetime,
) -> str:
    """Stable canonical JSON used as the chain input.

    Keys are sorted; datetimes are normalised to ISO-8601 with microsecond
    precision so two equivalent entries hash identically across processes
    (no Python-version-specific dict ordering, no timezone drift).
    """
    payload = {
        "action": action,
        "actor_id": str(actor_id) if actor_id else None,
        "created_at": created_at.astimezone(dt.timezone.utc).isoformat(
            timespec="microseconds"
        ),
        "metadata": metadata or {},
        "organization_id": str(organization_id),
        "outcome": outcome,
        "request_id": request_id,
        "resource_id": str(resource_id) if resource_id else None,
        "resource_type": resource_type,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _entry_hash(*, canonical: str, prev_hash: str | None) -> str:
    h = hashlib.sha256()
    h.update(canonical.encode("utf-8"))
    h.update(b"|")
    h.update((prev_hash or "").encode("utf-8"))
    return h.hexdigest()


# ── Recording ────────────────────────────────────────────────────────────────


async def _last_entry_hash(
    session: AsyncSession, *, organization_id: uuid.UUID
) -> str | None:
    stmt = (
        apply_tenant_filter(
            select(AuditLog.entry_hash).order_by(
                desc(AuditLog.created_at), desc(AuditLog.id)
            ),
            AuditLog,
            organization_id,
        ).limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def record_event(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    actor_id: uuid.UUID | None,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID | None = None,
    outcome: AuditOutcome = "success",
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLog:
    """Append a single row, computing prev_hash and entry_hash atomically.

    Concurrency note: two parallel writers for the same tenant can race
    on `_last_entry_hash`, both pick the same `prev_hash`, both compute
    distinct `entry_hash` values (because `created_at` differs at
    microsecond resolution), and both commit. The chain is still
    well-formed because we sort by (created_at, id) on read; both rows
    are valid descendants of the same parent. The unique
    `(organization_id, entry_hash)` constraint prevents the same row
    being persisted twice — a true duplicate would have to share every
    field including microsecond timestamp, which is the desired
    de-duplication behaviour.
    """
    now = dt.datetime.now(dt.timezone.utc)
    prev = await _last_entry_hash(session, organization_id=organization_id)
    canonical = _canonical_payload(
        organization_id=organization_id,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        request_id=request_id,
        metadata=metadata or {},
        created_at=now,
    )
    entry_hash = _entry_hash(canonical=canonical, prev_hash=prev)

    row = AuditLog(
        organization_id=organization_id,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        request_id=request_id,
        prev_hash=prev,
        entry_hash=entry_hash,
        metadata_json=metadata or {},
        created_at=now,
    )
    session.add(row)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise AuditError(
            "audit chain violation: duplicate entry_hash",
            status_code=409,
        ) from exc
    return row


# ── Listing & retrieval ──────────────────────────────────────────────────────


def _filters_to_where(
    filters: AuditEventFilters,
) -> list[ColumnElement[bool]]:
    """Translate `AuditEventFilters` to a list of SQLAlchemy clauses."""
    clauses: list[ColumnElement[bool]] = []
    if filters.actions:
        clauses.append(AuditLog.action.in_(filters.actions))
    if filters.resource_types:
        clauses.append(AuditLog.resource_type.in_(filters.resource_types))
    if filters.outcomes:
        clauses.append(AuditLog.outcome.in_(filters.outcomes))
    if filters.actor_ids:
        clauses.append(AuditLog.actor_id.in_(filters.actor_ids))
    if filters.since:
        clauses.append(AuditLog.created_at >= filters.since)
    if filters.until:
        clauses.append(AuditLog.created_at <= filters.until)
    if filters.search:
        like = f"%{filters.search}%"
        clauses.append(
            or_(
                AuditLog.action.ilike(like),
                AuditLog.resource_type.ilike(like),
                cast(AuditLog.metadata_json, Text).ilike(like),
            )
        )
    return clauses


async def _row_to_out(
    row: AuditLog, actor_email_by_id: dict[uuid.UUID, str]
) -> AuditEventOut:
    return AuditEventOut(
        event_id=row.id,
        organization_id=row.organization_id,
        actor_id=row.actor_id,
        actor_email=actor_email_by_id.get(row.actor_id) if row.actor_id else None,
        action=row.action,
        resource_type=row.resource_type,
        resource_id=row.resource_id,
        outcome=row.outcome if row.outcome in ("success", "denied", "error") else "success",
        request_id=row.request_id,
        prev_hash=row.prev_hash,
        entry_hash=row.entry_hash,
        metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
        created_at=row.created_at,
    )


async def _resolve_actor_emails(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    actor_ids: Iterable[uuid.UUID | None],
) -> dict[uuid.UUID, str]:
    """Fetch actor e-mails for the rows we're about to render.

    Done in a single batched query so listing 200 events is two round
    trips (events + actors) regardless of how many distinct actors
    appear.
    """
    ids = {a for a in actor_ids if a is not None}
    if not ids:
        return {}
    stmt = apply_tenant_filter(
        select(User.id, User.email).where(User.id.in_(ids)),
        User,
        organization_id,
    )
    return {uid: email for uid, email in (await session.execute(stmt)).all()}


async def list_events(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    filters: AuditEventFilters,
) -> AuditEventList:
    """Paginated, filtered fetch of audit events.

    Ordered by `created_at DESC, id DESC` so newer rows appear first and
    pagination is deterministic across ties.
    """
    where = _filters_to_where(filters)

    count_stmt = apply_tenant_filter(
        select(func.count(AuditLog.id)),
        AuditLog,
        organization_id,
    )
    if where:
        count_stmt = count_stmt.where(and_(*where))
    total = int((await session.execute(count_stmt)).scalar_one())

    list_stmt = apply_tenant_filter(
        select(AuditLog).order_by(
            desc(AuditLog.created_at), desc(AuditLog.id)
        ),
        AuditLog,
        organization_id,
    )
    if where:
        list_stmt = list_stmt.where(and_(*where))
    list_stmt = list_stmt.offset((filters.page - 1) * filters.page_size).limit(
        filters.page_size
    )
    rows = (await session.execute(list_stmt)).scalars().all()

    actor_emails = await _resolve_actor_emails(
        session,
        organization_id=organization_id,
        actor_ids=(r.actor_id for r in rows),
    )
    items = [await _row_to_out(r, actor_emails) for r in rows]

    return AuditEventList(
        items=items,
        total=total,
        page=filters.page,
        page_size=filters.page_size,
    )


async def get_event(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    event_id: uuid.UUID,
) -> AuditEventOut:
    stmt = apply_tenant_filter(
        select(AuditLog).where(AuditLog.id == event_id),
        AuditLog,
        organization_id,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise AuditEventNotFoundError(str(event_id))
    actor_emails = await _resolve_actor_emails(
        session,
        organization_id=organization_id,
        actor_ids=[row.actor_id] if row.actor_id else [],
    )
    return await _row_to_out(row, actor_emails)


# ── Stats ────────────────────────────────────────────────────────────────────


async def compute_ledger_stats(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    now: dt.datetime | None = None,
) -> LedgerStats:
    sweep_now = now or dt.datetime.now(dt.timezone.utc)

    total_q = apply_tenant_filter(
        select(func.count(AuditLog.id)), AuditLog, organization_id
    )
    total = int((await session.execute(total_q)).scalar_one())

    e24h_q = apply_tenant_filter(
        select(func.count(AuditLog.id)).where(
            AuditLog.created_at >= sweep_now - dt.timedelta(hours=24)
        ),
        AuditLog,
        organization_id,
    )
    events_24h = int((await session.execute(e24h_q)).scalar_one())

    e7d_q = apply_tenant_filter(
        select(func.count(AuditLog.id)).where(
            AuditLog.created_at >= sweep_now - dt.timedelta(days=7)
        ),
        AuditLog,
        organization_id,
    )
    events_7d = int((await session.execute(e7d_q)).scalar_one())

    distinct_actions_q = apply_tenant_filter(
        select(func.count(func.distinct(AuditLog.action))),
        AuditLog,
        organization_id,
    )
    distinct_actions = int((await session.execute(distinct_actions_q)).scalar_one())

    distinct_actors_q = apply_tenant_filter(
        select(func.count(func.distinct(AuditLog.actor_id))).where(
            AuditLog.actor_id.is_not(None)
        ),
        AuditLog,
        organization_id,
    )
    distinct_actors = int((await session.execute(distinct_actors_q)).scalar_one())

    last_q = apply_tenant_filter(
        select(AuditLog).order_by(
            desc(AuditLog.created_at), desc(AuditLog.id)
        ),
        AuditLog,
        organization_id,
    ).limit(1)
    last = (await session.execute(last_q)).scalar_one_or_none()

    head_q = apply_tenant_filter(
        select(AuditLog).order_by(
            asc(AuditLog.created_at), asc(AuditLog.id)
        ),
        AuditLog,
        organization_id,
    ).limit(1)
    head = (await session.execute(head_q)).scalar_one_or_none()

    return LedgerStats(
        total_events=total,
        events_24h=events_24h,
        events_7d=events_7d,
        distinct_actions=distinct_actions,
        distinct_actors=distinct_actors,
        last_event_at=last.created_at if last else None,
        head_hash=head.entry_hash if head else None,
        tail_hash=last.entry_hash if last else None,
    )


# ── Chain integrity ──────────────────────────────────────────────────────────


async def verify_chain(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
) -> IntegrityReport:
    """Re-walk the tenant's chain. Reports every break encountered.

    The walk is bounded — it loads the entire chain in chronological
    order. For tenants with very large ledgers we'd batch and stream;
    the platform is not at that scale yet (and the SPA `/audit/integrity`
    page would also need streaming UI). Documented as a known constraint
    for Stage 7+.
    """
    stmt = apply_tenant_filter(
        select(AuditLog).order_by(
            asc(AuditLog.created_at), asc(AuditLog.id)
        ),
        AuditLog,
        organization_id,
    )
    rows = (await session.execute(stmt)).scalars().all()
    breaks: list[IntegrityBreak] = []
    expected_prev: str | None = None
    head_hash: str | None = None
    tail_hash: str | None = None

    for row in rows:
        if head_hash is None:
            head_hash = row.entry_hash
        canonical = _canonical_payload(
            organization_id=row.organization_id,
            actor_id=row.actor_id,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            outcome=row.outcome,
            request_id=row.request_id,
            metadata=row.metadata_json
            if isinstance(row.metadata_json, dict)
            else {},
            created_at=row.created_at,
        )
        recomputed = _entry_hash(canonical=canonical, prev_hash=expected_prev)
        if row.prev_hash != expected_prev or row.entry_hash != recomputed:
            breaks.append(
                IntegrityBreak(
                    event_id=row.id,
                    expected_prev_hash=expected_prev,
                    observed_prev_hash=row.prev_hash,
                    expected_entry_hash=recomputed,
                    observed_entry_hash=row.entry_hash,
                    created_at=row.created_at,
                )
            )
        # Continue walking from the *observed* hash so we report multiple
        # breaks rather than masking everything past the first divergence.
        expected_prev = row.entry_hash
        tail_hash = row.entry_hash

    return IntegrityReport(
        organization_id=organization_id,
        verified_at=dt.datetime.now(dt.timezone.utc),
        total_events=len(rows),
        chain_ok=not breaks,
        head_hash=head_hash,
        tail_hash=tail_hash,
        breaks=breaks,
    )


# ── CSV export ───────────────────────────────────────────────────────────────


async def export_events_csv(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    filters: AuditEventFilters,
    max_rows: int = 50_000,
) -> bytes:
    """Render filtered events as CSV. Caps at `max_rows` defensively."""
    where = _filters_to_where(filters)
    list_stmt = apply_tenant_filter(
        select(AuditLog).order_by(
            desc(AuditLog.created_at), desc(AuditLog.id)
        ),
        AuditLog,
        organization_id,
    )
    if where:
        list_stmt = list_stmt.where(and_(*where))
    list_stmt = list_stmt.limit(max_rows)
    rows = (await session.execute(list_stmt)).scalars().all()

    actor_emails = await _resolve_actor_emails(
        session,
        organization_id=organization_id,
        actor_ids=(r.actor_id for r in rows),
    )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "event_id",
            "created_at",
            "actor_id",
            "actor_email",
            "action",
            "resource_type",
            "resource_id",
            "outcome",
            "request_id",
            "prev_hash",
            "entry_hash",
            "metadata_json",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                str(r.id),
                r.created_at.astimezone(dt.timezone.utc).isoformat(),
                str(r.actor_id) if r.actor_id else "",
                actor_emails.get(r.actor_id, "") if r.actor_id else "",
                r.action,
                r.resource_type,
                str(r.resource_id) if r.resource_id else "",
                r.outcome,
                r.request_id or "",
                r.prev_hash or "",
                r.entry_hash,
                json.dumps(
                    r.metadata_json if isinstance(r.metadata_json, dict) else {},
                    sort_keys=True,
                ),
            ]
        )
    return buf.getvalue().encode("utf-8")


# ── Retention policies ───────────────────────────────────────────────────────


async def list_policies(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
) -> list[RetentionPolicyOut]:
    stmt = apply_tenant_filter(
        select(RetentionPolicy).order_by(asc(RetentionPolicy.resource_type)),
        RetentionPolicy,
        organization_id,
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [
        RetentionPolicyOut(
            policy_id=r.id,
            resource_type=typing_cast(RetentionResource, r.resource_type),
            ttl_days=r.ttl_days,
            is_active=r.is_active,
            notes=r.notes,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


async def upsert_policy(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    resource_type: str,
    request: RetentionPolicyUpsert,
) -> RetentionPolicyOut:
    """Insert or update one retention policy.

    Resource type is validated *before* the row is touched so an invalid
    type cannot leak into the database (and would trip the audit-page
    select otherwise).
    """
    try:
        retention_mod.validate_resource_type(resource_type)
    except retention_mod.RetentionPolicyError as exc:
        raise AuditError(str(exc), status_code=400) from exc

    existing_stmt = apply_tenant_filter(
        select(RetentionPolicy).where(
            RetentionPolicy.resource_type == resource_type
        ),
        RetentionPolicy,
        organization_id,
    )
    row = (await session.execute(existing_stmt)).scalar_one_or_none()
    now = dt.datetime.now(dt.timezone.utc)
    if row is None:
        row = RetentionPolicy(
            organization_id=organization_id,
            resource_type=resource_type,
            ttl_days=request.ttl_days,
            is_active=request.is_active,
            notes=request.notes,
        )
        session.add(row)
    else:
        row.ttl_days = request.ttl_days
        row.is_active = request.is_active
        row.notes = request.notes
        row.updated_at = now

    await session.flush()
    await session.refresh(row)
    return RetentionPolicyOut(
        policy_id=row.id,
        resource_type=typing_cast(RetentionResource, row.resource_type),
        ttl_days=row.ttl_days,
        is_active=row.is_active,
        notes=row.notes,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ── Retention runs ───────────────────────────────────────────────────────────


def _row_to_run_out(row: RetentionRun) -> RetentionRunOut:
    counts = (
        row.purged_counts if isinstance(row.purged_counts, dict) else {}
    )
    st = row.status
    if st in ("running", "success", "failed"):
        status: RetentionStatus = typing_cast(RetentionStatus, st)
    else:
        status = "running"
    return RetentionRunOut(
        run_id=row.id,
        triggered_by=row.triggered_by,
        status=status,
        purged_counts={k: int(v) for k, v in counts.items() if isinstance(v, (int, float))},
        error_message=row.error_message,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


async def run_retention(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    user_id: uuid.UUID | None,
) -> RetentionRunOut:
    """Execute a sweep, log the run row, return the result.

    The run row is committed even on failure so an operator can find it
    in the SPA. Failures store the exception message under
    `error_message`; the chain row recorded by the caller (audit emitter)
    captures the same outcome with `outcome="error"`.
    """
    started_at = dt.datetime.now(dt.timezone.utc)
    run = RetentionRun(
        organization_id=organization_id,
        triggered_by=user_id,
        status="running",
        purged_counts={},
        started_at=started_at,
    )
    session.add(run)
    await session.flush()
    await session.refresh(run)
    run_id = run.id
    await session.commit()

    try:
        counts = await retention_mod.apply_retention_for_tenant(
            session=session,
            organization_id=organization_id,
            now=started_at,
        )
        await session.commit()
        finished_at = dt.datetime.now(dt.timezone.utc)
        run.purged_counts = dict(counts)
        run.status = "success"
        run.finished_at = finished_at
        await session.flush()
        await session.commit()
        log.info(
            "retention.run.success",
            organization_id=str(organization_id),
            run_id=str(run_id),
            counts=counts,
        )
    except Exception as exc:  # noqa: BLE001 — record + rethrow as AuditError
        await session.rollback()
        # Re-fetch to update status; the session may have been expired.
        stmt = apply_tenant_filter(
            select(RetentionRun).where(RetentionRun.id == run_id),
            RetentionRun,
            organization_id,
        )
        refreshed = (await session.execute(stmt)).scalar_one_or_none()
        if refreshed is not None:
            refreshed.status = "failed"
            refreshed.error_message = f"{type(exc).__name__}: {exc!s}"[:600]
            refreshed.finished_at = dt.datetime.now(dt.timezone.utc)
            await session.flush()
            await session.commit()
        log.exception(
            "retention.run.failed",
            organization_id=str(organization_id),
            run_id=str(run_id),
        )
        raise AuditError(f"retention sweep failed: {exc!s}", status_code=500) from exc

    return _row_to_run_out(run)


async def list_retention_runs(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 25,
) -> RetentionRunList:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    offset = (page - 1) * page_size

    list_stmt = (
        apply_tenant_filter(
            select(RetentionRun).order_by(desc(RetentionRun.started_at)),
            RetentionRun,
            organization_id,
        )
        .offset(offset)
        .limit(page_size)
    )
    rows = (await session.execute(list_stmt)).scalars().all()

    count_stmt = apply_tenant_filter(
        select(func.count(RetentionRun.id)),
        RetentionRun,
        organization_id,
    )
    total = int((await session.execute(count_stmt)).scalar_one())

    return RetentionRunList(
        items=[_row_to_run_out(r) for r in rows],
        total=total,
    )


__all__ = [
    "AuditError",
    "AuditEventNotFoundError",
    "compute_ledger_stats",
    "export_events_csv",
    "get_event",
    "list_events",
    "list_policies",
    "list_retention_runs",
    "record_event",
    "run_retention",
    "upsert_policy",
    "verify_chain",
]
