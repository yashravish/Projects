"""Lightweight emitter that fronts the audit service for service code.

The emitter:

  * **never raises** — if writing the audit row fails, the failure is
    logged and the calling business operation still returns success.
    A blocked audit write must not be allowed to take a tenant offline.
  * pulls the current request_id out of structlog's contextvars so the
    audit row joins back to the HTTP log line that created it.
  * delegates redaction of free-text payloads to `app.governance.pii`.

The emitter is intentionally a function, not a class. Stateless, no DI,
and trivially mockable in unit tests via `monkeypatch.setattr`.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from structlog.contextvars import get_contextvars

from app.governance import pii as pii_mod
from app.logging_config import get_logger
from app.schemas.audit import AuditOutcome
from app.services import audit_service

log = get_logger("audit_emitter")


def _current_request_id() -> str | None:
    ctx = get_contextvars()
    rid = ctx.get("request_id")
    return str(rid) if rid else None


async def emit(
    *,
    session: AsyncSession,
    organization_id: uuid.UUID,
    actor_id: uuid.UUID | None,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID | None = None,
    outcome: AuditOutcome = "success",
    metadata: dict[str, Any] | None = None,
    request_id: str | None = None,
    commit: bool = True,
) -> None:
    """Write an audit row. Never raises.

    `commit=True` (the default) commits the audit row in its own
    transaction so it persists even if the caller's later work fails.
    Pass `commit=False` when calling from inside a transaction the
    caller will commit anyway (the audit row stays in the same unit
    of work).
    """
    rid = request_id if request_id is not None else _current_request_id()
    try:
        await audit_service.record_event(
            session=session,
            organization_id=organization_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            request_id=rid,
            metadata=metadata or {},
        )
        if commit:
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — emitter must not raise
        log.warning(
            "audit.emit.failed",
            organization_id=str(organization_id),
            action=action,
            resource_type=resource_type,
            error=str(exc),
        )
        try:
            await session.rollback()
        except Exception:  # noqa: BLE001
            log.exception("audit.emit.rollback_failed")


def redact_text_for_audit(text: str, *, max_length: int = 200) -> dict[str, Any]:
    """Run PII detection, return a redaction-safe summary for metadata.

    Returns a dict suitable for stuffing into the audit metadata column:

        {
          "preview": "What's the budget for [REDACTED:email] dept?",
          "pii_kinds": {"email": 1},
          "had_pii": true,
          "n_chars": 47,
        }
    """
    if not text:
        return {"preview": "", "pii_kinds": {}, "had_pii": False, "n_chars": 0}
    redacted, findings = pii_mod.redact(text)
    preview = redacted[:max_length]
    if len(redacted) > max_length:
        preview = preview + "…"
    return {
        "preview": preview,
        "pii_kinds": pii_mod.summarize(findings),
        "had_pii": bool(findings),
        "n_chars": len(text),
    }


__all__ = ["emit", "redact_text_for_audit"]
