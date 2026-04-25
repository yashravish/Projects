"""End-to-end /query/inquiry test.

Drives:
- registration → authenticated session
- one PDF upload + synchronous ingestion (no Celery)
- POST /api/v1/query/inquiry → assert status 200, citations bound, trace
- GET /api/v1/query/runs → assert the run shows up
- GET /api/v1/query/runs/{id} → assert detail replays the answer

Uses the same offline LLM + LocalDeterministicEmbedder stack the dev server
runs in without an OPENAI_API_KEY, so this passes deterministically in CI.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import requires_postgres

pytestmark = [pytest.mark.asyncio, requires_postgres]


def _register_payload(suffix: str) -> dict[str, str]:
    """Per-test unique credentials so the integration tests don't collide on
    the seed DB (which the `client` fixture leaves intact)."""
    return {
        "email": f"inquiry-{suffix}@example.gov",
        "password": "AnalystPass!2026",
        "organization_name": f"Inquiry Test Agency {suffix}",
    }


async def _register(client, *, suffix: str) -> str:
    r = await client.post("/api/v1/auth/register", json=_register_payload(suffix))
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


async def _upload_and_ingest(client, *, token: str, sample_index: int = 0) -> str:
    """Upload one of the synthetic seed PDFs and run ingestion synchronously."""
    from app.agents.llm_client import build_embedder
    from app.db.session import async_session_factory
    from app.seed.generate_sample_pdfs import build_sample_pdfs
    from app.services import ingestion_service
    from app.services.storage_service import build_default_scanner, get_storage

    sample = build_sample_pdfs()[sample_index]
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (sample.filename, sample.bytes_, "application/pdf")}
    r = await client.post("/api/v1/documents/upload", headers=headers, files=files)
    assert r.status_code == 201, r.text
    doc_id = r.json()["document_id"]

    storage = get_storage()
    embedder = build_embedder()
    scanner = build_default_scanner()  # noqa: F841 — kept for symmetry / future hooks
    async with async_session_factory() as session:
        from sqlalchemy import select

        from app.db.models import Document

        doc = (
            (await session.execute(select(Document).where(Document.id == doc_id)))
            .scalar_one()
        )
        await ingestion_service.ingest_document(
            session,
            organization_id=doc.organization_id,
            document_id=doc.id,
            storage=storage,
            embedder=embedder,
        )
    return doc_id


@pytest.mark.asyncio
async def test_inquiry_end_to_end(client) -> None:
    token = await _register(client, suffix="e2e")
    headers = {"Authorization": f"Bearer {token}"}

    # Upload all three sample PDFs so retrieval has real material to fuse.
    for i in range(3):
        await _upload_and_ingest(client, token=token, sample_index=i)

    inquiry = {"question": "What is the deadline for the Resilient Communities grant?"}
    r = await client.post("/api/v1/query/inquiry", headers=headers, json=inquiry)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()

    assert body["status"] == "success"
    assert body["question"] == inquiry["question"]
    assert body["answer_text"]
    assert body["model"]
    assert body["latency_ms"] >= 0
    # Trace should include all four nodes.
    nodes = [t["node"] for t in body["trace"]]
    assert nodes == ["plan", "retrieve", "synthesize", "critique"]
    # Some retrieval should have happened.
    assert body["retrieved"], "expected at least one retrieved chunk"
    # The offline synthesizer always quotes [1], so we expect ≥1 citation.
    assert body["citations"], "expected at least one citation"
    # Each citation must reference one of the retrieved chunks.
    retrieved_ids = {c["chunk_id"] for c in body["retrieved"]}
    for c in body["citations"]:
        assert c["chunk_id"] in retrieved_ids
    # Critique fields are well-formed.
    crit = body["critique"]
    assert 0.0 <= crit["grounding_score"] <= 1.0
    assert 0.0 <= crit["hallucination_risk"] <= 1.0

    # Run shows up in the listing.
    list_r = await client.get("/api/v1/query/runs", headers=headers)
    assert list_r.status_code == 200, list_r.text
    runs = list_r.json()
    assert runs["total"] >= 1
    run_ids = [item["run_id"] for item in runs["items"]]
    assert body["run_id"] in run_ids

    # Detail endpoint replays the run from JSONB.
    detail_r = await client.get(
        f"/api/v1/query/runs/{body['run_id']}", headers=headers
    )
    assert detail_r.status_code == 200, detail_r.text
    detail = detail_r.json()
    assert detail["answer_text"] == body["answer_text"]
    assert len(detail["citations"]) == len(body["citations"])


@pytest.mark.asyncio
async def test_inquiry_rejects_short_question(client) -> None:
    token = await _register(client, suffix="short")
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.post(
        "/api/v1/query/inquiry", headers=headers, json={"question": "Hi"}
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_inquiry_run_is_tenant_scoped(client) -> None:
    """Org A's run must be invisible to org B's authenticated user."""
    token_a = await _register(client, suffix="iso-a")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    await _upload_and_ingest(client, token=token_a, sample_index=0)
    r = await client.post(
        "/api/v1/query/inquiry",
        headers=headers_a,
        json={"question": "What does the resilient communities grant fund?"},
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]

    token_b = await _register(client, suffix="iso-b")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Detail lookup for org A's run from org B must 404.
    detail_r = await client.get(f"/api/v1/query/runs/{run_id}", headers=headers_b)
    assert detail_r.status_code == 404
    # Listing for org B must not include org A's run.
    list_r = await client.get("/api/v1/query/runs", headers=headers_b)
    assert list_r.status_code == 200
    other_run_ids = [item["run_id"] for item in list_r.json()["items"]]
    assert run_id not in other_run_ids
