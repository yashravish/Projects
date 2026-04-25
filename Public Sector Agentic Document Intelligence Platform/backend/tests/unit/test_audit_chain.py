"""Unit tests for the canonicalisation + entry-hash machinery.

The chain function only needs raw inputs — no DB. This isolates the
correctness of the hash math from any persistence concerns.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

from app.services.audit_service import _canonical_payload, _entry_hash


def _sample_kwargs(**overrides):
    base = dict(
        organization_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        actor_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        action="document.upload",
        resource_type="document",
        resource_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        outcome="success",
        request_id="req-abc",
        metadata={"k": "v", "n": 3},
        created_at=dt.datetime(2026, 4, 25, 12, 0, 0, 123456, tzinfo=dt.timezone.utc),
    )
    base.update(overrides)
    return base


def test_canonical_payload_is_stable_across_metadata_key_order() -> None:
    a = _canonical_payload(**_sample_kwargs(metadata={"alpha": 1, "beta": 2}))
    b = _canonical_payload(**_sample_kwargs(metadata={"beta": 2, "alpha": 1}))
    assert a == b
    # And the JSON is parseable round-trip.
    assert json.loads(a)["metadata"] == {"alpha": 1, "beta": 2}


def test_entry_hash_changes_when_any_field_changes() -> None:
    base = _sample_kwargs()
    base_hash = _entry_hash(canonical=_canonical_payload(**base), prev_hash=None)

    # Mutating each field independently changes the hash.
    for override in [
        {"action": "document.delete"},
        {"actor_id": uuid.UUID("44444444-4444-4444-4444-444444444444")},
        {"outcome": "denied"},
        {"request_id": "req-xyz"},
        {"metadata": {"k": "different"}},
    ]:
        h = _entry_hash(
            canonical=_canonical_payload(**_sample_kwargs(**override)),
            prev_hash=None,
        )
        assert h != base_hash, f"hash unchanged for override {override}"


def test_entry_hash_changes_when_prev_hash_changes() -> None:
    base = _sample_kwargs()
    canonical = _canonical_payload(**base)
    h_none = _entry_hash(canonical=canonical, prev_hash=None)
    h_a = _entry_hash(canonical=canonical, prev_hash="a" * 64)
    h_b = _entry_hash(canonical=canonical, prev_hash="b" * 64)
    assert h_none != h_a
    assert h_a != h_b


def test_entry_hash_is_deterministic() -> None:
    canonical = _canonical_payload(**_sample_kwargs())
    a = _entry_hash(canonical=canonical, prev_hash="abc")
    b = _entry_hash(canonical=canonical, prev_hash="abc")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_canonical_payload_normalises_timezones() -> None:
    """Two equivalent timestamps in different zones produce the same payload."""
    utc = dt.datetime(2026, 4, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
    other = utc.astimezone(dt.timezone(dt.timedelta(hours=-5)))
    a = _canonical_payload(**_sample_kwargs(created_at=utc))
    b = _canonical_payload(**_sample_kwargs(created_at=other))
    assert a == b
