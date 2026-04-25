"""Pydantic shapes for the audit & governance API.

The audit log is read-mostly: it has many filtering and listing endpoints
but only a handful of write endpoints (retention runs, policy upserts).
The integrity report is a separate type from the regular event because
its shape is operational rather than per-event.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AuditOutcome = Literal["success", "denied", "error"]
RetentionResource = Literal["document", "query_run", "evaluation_run"]
RetentionStatus = Literal["running", "success", "failed"]


# ── Events ──────────────────────────────────────────────────────────────────


class AuditEventOut(BaseModel):
    """One audit ledger entry, fully expanded for the SPA / CSV export.

    `entry_hash` and `prev_hash` are exposed so the frontend can render a
    chain badge on the detail drawer; the integrity verification still
    happens server-side.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID
    organization_id: uuid.UUID
    actor_id: uuid.UUID | None
    actor_email: str | None
    action: str
    resource_type: str
    resource_id: uuid.UUID | None
    outcome: AuditOutcome
    request_id: str | None
    prev_hash: str | None
    entry_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: dt.datetime


class AuditEventList(BaseModel):
    """Paginated list response."""

    model_config = ConfigDict(extra="forbid")

    items: list[AuditEventOut]
    total: int
    page: int
    page_size: int


class AuditEventFilters(BaseModel):
    """Query-string filters for `GET /audit/events`."""

    model_config = ConfigDict(extra="forbid")

    actions: list[str] | None = None
    resource_types: list[str] | None = None
    outcomes: list[AuditOutcome] | None = None
    actor_ids: list[uuid.UUID] | None = None
    since: dt.datetime | None = None
    until: dt.datetime | None = None
    search: str | None = Field(default=None, max_length=200)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


# ── Integrity ───────────────────────────────────────────────────────────────


class IntegrityBreak(BaseModel):
    """A specific row where the chain failed verification."""

    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID
    expected_prev_hash: str | None
    observed_prev_hash: str | None
    expected_entry_hash: str
    observed_entry_hash: str
    created_at: dt.datetime


class IntegrityReport(BaseModel):
    """End-to-end chain verification result for one tenant."""

    model_config = ConfigDict(extra="forbid")

    organization_id: uuid.UUID
    verified_at: dt.datetime
    total_events: int
    chain_ok: bool
    head_hash: str | None
    tail_hash: str | None
    breaks: list[IntegrityBreak] = Field(default_factory=list)


# ── Retention ───────────────────────────────────────────────────────────────


class RetentionPolicyOut(BaseModel):
    """One per (org, resource_type)."""

    model_config = ConfigDict(extra="forbid")

    policy_id: uuid.UUID
    resource_type: RetentionResource
    ttl_days: int
    is_active: bool
    notes: str | None
    created_at: dt.datetime
    updated_at: dt.datetime


class RetentionPolicyUpsert(BaseModel):
    """Body for `PUT /audit/policies/{resource_type}`."""

    model_config = ConfigDict(extra="forbid")

    ttl_days: int = Field(ge=0, le=36500)
    is_active: bool = True
    notes: str | None = Field(default=None, max_length=2_000)


class RetentionPolicyList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RetentionPolicyOut]


class RetentionRunOut(BaseModel):
    """Audit of one executed sweep."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    triggered_by: uuid.UUID | None
    status: RetentionStatus
    purged_counts: dict[str, int]
    error_message: str | None
    started_at: dt.datetime
    finished_at: dt.datetime | None


class RetentionRunList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RetentionRunOut]
    total: int


# ── Stats (header card on the Ledger) ───────────────────────────────────────


class LedgerStats(BaseModel):
    """High-level counts for the Ledger header strip."""

    model_config = ConfigDict(extra="forbid")

    total_events: int
    events_24h: int
    events_7d: int
    distinct_actions: int
    distinct_actors: int
    last_event_at: dt.datetime | None
    head_hash: str | None
    tail_hash: str | None


__all__ = [
    "AuditEventFilters",
    "AuditEventList",
    "AuditEventOut",
    "AuditOutcome",
    "IntegrityBreak",
    "IntegrityReport",
    "LedgerStats",
    "RetentionPolicyList",
    "RetentionPolicyOut",
    "RetentionPolicyUpsert",
    "RetentionResource",
    "RetentionRunList",
    "RetentionRunOut",
    "RetentionStatus",
]
