"""End-to-end /evaluations API test.

Drives:
- registration → authenticated session
- ingest all three seed PDFs synchronously (no Celery)
- GET /api/v1/evaluations/dataset → assert the gold set is well-formed
- POST /api/v1/evaluations/run → assert 200, aggregate present, items match dataset
- GET /api/v1/evaluations → assert the run shows up in the listing
- GET /api/v1/evaluations/{id} → assert detail replays
- Tenant isolation: another org's user cannot see the run

Uses the offline LLM + LocalDeterministicEmbedder so this is deterministic in CI.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import requires_postgres
from tests.integration.test_query_flow import _register, _upload_and_ingest

pytestmark = [pytest.mark.asyncio, requires_postgres]


async def _seed_corpus_for_token(client, *, token: str) -> None:
    for i in range(3):
        await _upload_and_ingest(client, token=token, sample_index=i)


@pytest.mark.asyncio
async def test_dataset_endpoint_is_well_formed(client) -> None:
    token = await _register(client, suffix="eval-ds")
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.get("/api/v1/evaluations/dataset", headers=headers)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    assert body["name"]
    assert body["version"]
    assert body["n_items"] >= 5
    assert isinstance(body["items"], list)
    # Each item carries a unique id and at least one expected document.
    ids = [it["id"] for it in body["items"]]
    assert len(ids) == len(set(ids))
    for item in body["items"]:
        assert item["expected_doc_filenames"]
        assert item["must_contain_any"]


@pytest.mark.asyncio
async def test_evaluation_run_end_to_end(client) -> None:
    token = await _register(client, suffix="eval-e2e")
    headers = {"Authorization": f"Bearer {token}"}
    await _seed_corpus_for_token(client, token=token)

    r = await client.post(
        "/api/v1/evaluations/run", headers=headers, json={}
    )
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()

    assert body["status"] == "success"
    assert body["dataset_name"]
    assert body["dataset_version"]
    assert body["aggregate"]["n_items"] == len(body["items"])
    assert body["aggregate"]["n_items"] >= 5
    # Aggregate fields are well-formed numbers.
    agg = body["aggregate"]
    for key in (
        "pass_rate",
        "retrieval_recall",
        "retrieval_precision",
        "citation_precision",
        "citation_recall",
        "faithfulness",
        "forbidden_phrase_rate",
        "grounding_score",
        "hallucination_risk",
        "latency_ms_p50",
        "latency_ms_p95",
    ):
        assert isinstance(agg[key], (int, float)), key
        assert 0.0 <= agg[key] or key.startswith("latency_ms"), (
            f"{key} should be non-negative"
        )

    # Per-item structure: gold + metrics + inquiry.
    seen_ids: set[str] = set()
    for it in body["items"]:
        assert it["gold"]["id"]
        assert it["gold"]["id"] not in seen_ids
        seen_ids.add(it["gold"]["id"])
        assert it["metrics"]["item_id"] == it["gold"]["id"]
        assert it["inquiry"]["question"] == it["gold"]["question"]
        assert "answer_text" in it["inquiry"]

    # The local-deterministic embedder is weak; we don't assert pass_rate > 0.
    # But at least one retrieval recall should be > 0 across the dataset.
    assert any(
        it["metrics"]["retrieval_recall"] > 0 for it in body["items"]
    ), "no item retrieved an expected doc — retrieval is broken"

    # Listing endpoint sees the new run.
    list_r = await client.get("/api/v1/evaluations", headers=headers)
    assert list_r.status_code == 200, list_r.text
    listing = list_r.json()
    assert listing["total"] >= 1
    run_ids = [item["run_id"] for item in listing["items"]]
    assert body["run_id"] in run_ids

    # Detail endpoint replays.
    detail_r = await client.get(
        f"/api/v1/evaluations/{body['run_id']}", headers=headers
    )
    assert detail_r.status_code == 200, detail_r.text
    detail = detail_r.json()
    assert detail["aggregate"]["n_items"] == body["aggregate"]["n_items"]
    assert len(detail["items"]) == len(body["items"])


@pytest.mark.asyncio
async def test_evaluation_run_is_tenant_scoped(client) -> None:
    token_a = await _register(client, suffix="eval-iso-a")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    await _upload_and_ingest(client, token=token_a, sample_index=0)
    r = await client.post(
        "/api/v1/evaluations/run", headers=headers_a, json={}
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]

    token_b = await _register(client, suffix="eval-iso-b")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    detail_r = await client.get(
        f"/api/v1/evaluations/{run_id}", headers=headers_b
    )
    assert detail_r.status_code == 404

    list_r = await client.get("/api/v1/evaluations", headers=headers_b)
    assert list_r.status_code == 200
    visible = [item["run_id"] for item in list_r.json()["items"]]
    assert run_id not in visible


@pytest.mark.asyncio
async def test_evaluation_unknown_dataset_returns_400(client) -> None:
    token = await _register(client, suffix="eval-bad-ds")
    headers = {"Authorization": f"Bearer {token}"}
    r = await client.post(
        "/api/v1/evaluations/run",
        headers=headers,
        json={"dataset_name": "nonsense"},
    )
    assert r.status_code == 400
