"""End-to-end integration tests for the audit ledger and retention.

Covers:

  * register/login emit audit rows visible at /audit/events
  * the chain verifies clean
  * tampering with a row breaks the chain at exactly that row
  * tenant isolation: tenant A's events are invisible to tenant B
  * retention policy upsert + sweep + audit row of the sweep itself
  * CSV export returns text/csv with the right header

The slow marker is set so this can be opted out of during unit-only runs.
"""
from __future__ import annotations


import pytest
from httpx import AsyncClient
from sqlalchemy import select, update

from tests.conftest import requires_postgres

pytestmark = [requires_postgres, pytest.mark.asyncio, pytest.mark.slow]


async def _register(client: AsyncClient, *, email: str, org: str) -> dict:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "AnalystPass!2026",
            "organization_name": org,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_register_and_login_produce_audit_rows(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="alice@example.gov", org="Agency Alpha")
    token = body["access_token"]

    # Drive a login to add a second event.
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.gov", "password": "AnalystPass!2026"},
    )
    assert login.status_code == 200

    # And a failed login (wrong password) — must record outcome=denied.
    bad = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.gov", "password": "wrong-password-12345"},
    )
    assert bad.status_code == 401

    listing = await client.get("/api/v1/audit/events", headers=_auth_headers(token))
    assert listing.status_code == 200, listing.text
    data = listing.json()
    actions = [e["action"] for e in data["items"]]
    outcomes = {(e["action"], e["outcome"]) for e in data["items"]}

    assert "auth.register" in actions
    assert ("auth.login", "success") in outcomes
    assert ("auth.login", "denied") in outcomes
    assert data["total"] >= 3


async def test_chain_verifies_clean(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="bob@example.gov", org="Agency Beta")
    token = body["access_token"]

    integrity = await client.get(
        "/api/v1/audit/integrity", headers=_auth_headers(token)
    )
    assert integrity.status_code == 200
    report = integrity.json()
    assert report["chain_ok"] is True
    assert report["breaks"] == []
    assert report["total_events"] >= 1


async def test_tamper_breaks_chain(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="carol@example.gov", org="Agency Gamma")
    token = body["access_token"]

    # Drive at least three events so we can mutate a middle row.
    await client.post(
        "/api/v1/auth/login",
        json={"email": "carol@example.gov", "password": "AnalystPass!2026"},
    )
    await client.post(
        "/api/v1/auth/login",
        json={"email": "carol@example.gov", "password": "wrong-password-1"},
    )

    from app.db.models import AuditLog

    # Mutate the action of one row in place — this should break the chain.
    rows = (
        await db_session.execute(
            select(AuditLog).order_by(AuditLog.created_at.asc())
        )
    ).scalars().all()
    assert len(rows) >= 2
    target = rows[len(rows) // 2]
    await db_session.execute(
        update(AuditLog)
        .where(AuditLog.id == target.id)
        .values(action="auth.login.TAMPERED")
    )
    await db_session.commit()

    integrity = await client.get(
        "/api/v1/audit/integrity", headers=_auth_headers(token)
    )
    report = integrity.json()
    assert report["chain_ok"] is False
    assert len(report["breaks"]) >= 1
    # The first reported break is at or after the tampered row.
    break_ids = [b["event_id"] for b in report["breaks"]]
    assert str(target.id) in break_ids


async def test_tenant_isolation_for_events(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    a = await _register(client, email="dan@example.gov", org="Agency Delta")
    b = await _register(client, email="erin@example.gov", org="Agency Epsilon")

    a_events = await client.get(
        "/api/v1/audit/events",
        headers=_auth_headers(a["access_token"]),
    )
    b_events = await client.get(
        "/api/v1/audit/events",
        headers=_auth_headers(b["access_token"]),
    )
    assert a_events.status_code == 200
    assert b_events.status_code == 200
    a_ids = {e["event_id"] for e in a_events.json()["items"]}
    b_ids = {e["event_id"] for e in b_events.json()["items"]}
    assert a_ids and b_ids
    assert a_ids.isdisjoint(b_ids)


async def test_retention_policy_upsert_and_sweep(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="frank@example.gov", org="Agency Zeta")
    token = body["access_token"]
    headers = _auth_headers(token)

    upsert = await client.put(
        "/api/v1/audit/policies/query_run",
        headers=headers,
        json={"ttl_days": 30, "is_active": True, "notes": "30d for query runs"},
    )
    assert upsert.status_code == 200, upsert.text
    policy = upsert.json()
    assert policy["resource_type"] == "query_run"
    assert policy["ttl_days"] == 30

    listing = await client.get("/api/v1/audit/policies", headers=headers)
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(p["resource_type"] == "query_run" for p in items)

    # Audit_log policy is forbidden.
    forbidden = await client.put(
        "/api/v1/audit/policies/audit_log",
        headers=headers,
        json={"ttl_days": 7, "is_active": True},
    )
    assert forbidden.status_code == 400

    run = await client.post(
        "/api/v1/audit/retention/runs", headers=headers
    )
    assert run.status_code == 200, run.text
    sweep = run.json()
    assert sweep["status"] == "success"
    assert isinstance(sweep["purged_counts"], dict)

    history = await client.get(
        "/api/v1/audit/retention/runs", headers=headers
    )
    assert history.status_code == 200
    assert history.json()["total"] >= 1


async def test_csv_export(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="gail@example.gov", org="Agency Eta")
    token = body["access_token"]
    csv_resp = await client.get(
        "/api/v1/audit/events.csv", headers=_auth_headers(token)
    )
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    text = csv_resp.text
    assert text.startswith("event_id,created_at,actor_id,")
    assert "auth.register" in text


async def test_stats_endpoint(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    body = await _register(client, email="hank@example.gov", org="Agency Theta")
    token = body["access_token"]
    resp = await client.get(
        "/api/v1/audit/stats", headers=_auth_headers(token)
    )
    assert resp.status_code == 200
    stats = resp.json()
    assert stats["total_events"] >= 1
    assert stats["events_24h"] >= 1
    assert stats["head_hash"]
    assert stats["tail_hash"]
