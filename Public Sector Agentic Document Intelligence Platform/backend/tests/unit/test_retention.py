"""Unit tests for `app.governance.retention`.

`apply_retention_for_tenant` is async and operates on a SQLAlchemy
session, so the deeper behaviour is tested in the integration suite.
Here we cover the synchronous validation helpers and the cutoff math.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.governance.retention import (
    PROHIBITED_RESOURCE_TYPES,
    SUPPORTED_RESOURCE_TYPES,
    RetentionPolicyError,
    _resolve_cutoff,
    validate_resource_type,
)


def test_supported_types_are_disjoint_from_prohibited() -> None:
    assert set(SUPPORTED_RESOURCE_TYPES).isdisjoint(set(PROHIBITED_RESOURCE_TYPES))


def test_validate_accepts_supported() -> None:
    for rt in SUPPORTED_RESOURCE_TYPES:
        validate_resource_type(rt)  # must not raise


def test_validate_rejects_audit_log_for_retention() -> None:
    with pytest.raises(RetentionPolicyError) as exc_info:
        validate_resource_type("audit_log")
    assert "audit ledger" in str(exc_info.value).lower()


def test_validate_rejects_unknown() -> None:
    with pytest.raises(RetentionPolicyError):
        validate_resource_type("definitely_not_a_resource")


def test_resolve_cutoff_subtracts_days() -> None:
    now = dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
    cutoff = _resolve_cutoff(7, now=now)
    assert cutoff == dt.datetime(2026, 4, 18, 12, 0, 0, tzinfo=dt.timezone.utc)


def test_resolve_cutoff_zero_returns_now() -> None:
    now = dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
    assert _resolve_cutoff(0, now=now) == now
