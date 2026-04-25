"""Per-tenant data retention enforcement.

A retention sweep walks the active `RetentionPolicy` rows for one
organization, computes a cutoff timestamp from each policy's `ttl_days`,
and purges resources older than that cutoff. The function returns the
counts so the caller (the audit page, an operator script, or a future
scheduled Celery task) can show what was removed.

Resource types currently supported:

  * `query_run`        — hard-deletes `query_runs` rows past TTL
  * `evaluation_run`   — hard-deletes `evaluation_runs` rows past TTL
  * `document`         — soft-deletes `documents` (sets `deleted_at`) past TTL.
                          Chunks are removed via the FK cascade.
  * `audit_log`        — IGNORED. The audit ledger is intentionally
                          immune to retention; a policy targeting it
                          raises a `RetentionPolicyError`.

`audit_log` is excluded by design: a tamper-evident chain that can be
truncated by a routine purge isn't tamper-evident. If you need to age
out very old audit rows, do it under a separate, manually-triggered
'archive' migration that exports first, with chain-tail proofs preserved.

The function never raises on a missing policy or zero-TTL policy; both
cases are no-ops. It DOES raise `RetentionPolicyError` for an invalid
resource type so misconfiguration is surfaced loudly at the API layer.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AuditLog,
    Document,
    EvaluationRun,
    QueryRun,
    RetentionPolicy,
)
from app.security.tenant import apply_tenant_filter

SUPPORTED_RESOURCE_TYPES: tuple[str, ...] = (
    "document",
    "query_run",
    "evaluation_run",
)
PROHIBITED_RESOURCE_TYPES: tuple[str, ...] = ("audit_log",)


class RetentionPolicyError(Exception):
    """Raised when a retention policy targets an unsupported resource type."""


def _resolve_cutoff(ttl_days: int, now: dt.datetime | None = None) -> dt.datetime:
    base = now or dt.datetime.now(dt.timezone.utc)
    return base - dt.timedelta(days=ttl_days)


async def _purge_query_runs(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    cutoff: dt.datetime,
) -> int:
    stmt = apply_tenant_filter(
        select(QueryRun.id).where(QueryRun.created_at < cutoff),
        QueryRun,
        organization_id,
    )
    ids = [row for row in (await session.execute(stmt)).scalars().all()]
    if not ids:
        return 0
    await session.execute(delete(QueryRun).where(QueryRun.id.in_(ids)))
    return len(ids)


async def _purge_evaluation_runs(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    cutoff: dt.datetime,
) -> int:
    stmt = apply_tenant_filter(
        select(EvaluationRun.id).where(EvaluationRun.created_at < cutoff),
        EvaluationRun,
        organization_id,
    )
    ids = [row for row in (await session.execute(stmt)).scalars().all()]
    if not ids:
        return 0
    await session.execute(delete(EvaluationRun).where(EvaluationRun.id.in_(ids)))
    return len(ids)


async def _soft_delete_documents(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    cutoff: dt.datetime,
    now: dt.datetime,
) -> int:
    stmt = apply_tenant_filter(
        select(Document.id)
        .where(Document.created_at < cutoff)
        .where(Document.deleted_at.is_(None)),
        Document,
        organization_id,
    )
    ids = [row for row in (await session.execute(stmt)).scalars().all()]
    if not ids:
        return 0
    await session.execute(
        update(Document).where(Document.id.in_(ids)).values(deleted_at=now)
    )
    return len(ids)


# ── Public API ───────────────────────────────────────────────────────────────


async def apply_retention_for_tenant(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    now: dt.datetime | None = None,
) -> dict[str, int]:
    """Run every active retention policy for the tenant. Returns purged counts.

    Counts include a `0` for resource types whose policy is active but
    matched no rows. Resource types without an active policy are not
    reported. `audit_log` is silently skipped if a policy slipped through
    for it via direct DB write — the API layer rejects it earlier.
    """
    sweep_now = now or dt.datetime.now(dt.timezone.utc)

    stmt = apply_tenant_filter(
        select(RetentionPolicy).where(RetentionPolicy.is_active.is_(True)),
        RetentionPolicy,
        organization_id,
    )
    policies = (await session.execute(stmt)).scalars().all()

    counts: dict[str, int] = {}
    for p in policies:
        if p.resource_type in PROHIBITED_RESOURCE_TYPES:
            continue
        if p.ttl_days <= 0:
            continue
        cutoff = _resolve_cutoff(p.ttl_days, now=sweep_now)
        if p.resource_type == "query_run":
            counts["query_run"] = await _purge_query_runs(
                session,
                organization_id=organization_id,
                cutoff=cutoff,
            )
        elif p.resource_type == "evaluation_run":
            counts["evaluation_run"] = await _purge_evaluation_runs(
                session,
                organization_id=organization_id,
                cutoff=cutoff,
            )
        elif p.resource_type == "document":
            counts["document"] = await _soft_delete_documents(
                session,
                organization_id=organization_id,
                cutoff=cutoff,
                now=sweep_now,
            )
        # Unknown resource_type: ignore silently — `validate_resource_type`
        # guards the write path; old rows remaining shouldn't break a sweep.
    return counts


def validate_resource_type(resource_type: str) -> None:
    """Raise RetentionPolicyError for unsupported / prohibited types."""
    if resource_type in PROHIBITED_RESOURCE_TYPES:
        raise RetentionPolicyError(
            f"resource type {resource_type!r} cannot have a retention policy "
            "(audit ledger is immutable)"
        )
    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        raise RetentionPolicyError(
            f"unsupported resource type {resource_type!r}; "
            f"valid: {', '.join(SUPPORTED_RESOURCE_TYPES)}"
        )


def audit_log_count_view(audit_log_cls: type[AuditLog]) -> type[AuditLog]:
    """Re-export so callers don't need to import models for stat queries.

    Used by the audit service to expose a tenant audit-row count alongside
    the most recent retention sweep, without coupling the service file to
    the SQLAlchemy import graph.
    """
    return audit_log_cls


__all__ = [
    "PROHIBITED_RESOURCE_TYPES",
    "RetentionPolicyError",
    "SUPPORTED_RESOURCE_TYPES",
    "apply_retention_for_tenant",
    "validate_resource_type",
]
