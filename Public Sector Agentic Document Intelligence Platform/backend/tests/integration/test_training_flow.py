"""End-to-end /training + /models API test.

Drives:
  - register → authenticated session
  - POST /api/v1/training/jobs (auto-promote=True) → status 200, success, model registered
  - GET /api/v1/training/jobs → the run shows up
  - GET /api/v1/training/jobs/{id} → detail replays + carries metrics
  - GET /api/v1/models → at least one production model
  - GET /api/v1/models/{id} → detail
  - POST /api/v1/models/{id}/predict → scores in expected order
  - POST /api/v1/models/{id}/promote { stage: archived } → row moves
  - Tenant isolation: another org cannot see / predict the model

Uses the LocalTrainingBackend → real subprocess. Slow (~3-5 s) but the only
way to catch packaging regressions in `app.ml.training_script`.
"""
from __future__ import annotations

import pytest

from tests.conftest import requires_postgres
from tests.integration.test_query_flow import _register

pytestmark = [pytest.mark.asyncio, requires_postgres, pytest.mark.slow]


@pytest.mark.asyncio
async def test_training_end_to_end(client) -> None:
    token = await _register(client, suffix="train-e2e")
    headers = {"Authorization": f"Bearer {token}"}

    # Submit
    r = await client.post(
        "/api/v1/training/jobs",
        headers=headers,
        json={
            "name": "psdi-cross-encoder-reranker",
            "auto_promote": True,
            "notes": "integration test run",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body.get("error_message") or body
    assert body["backend"] in ("local", "sagemaker")
    assert body["framework"] == "sklearn-tfidf-logreg"
    assert body["registered_model_id"], "auto_promote should produce a registered model"
    assert body["metrics"]["holdout_f1"] >= 0.0
    job_id = body["job_id"]
    model_id = body["registered_model_id"]

    # Listing carries the job
    list_r = await client.get("/api/v1/training/jobs", headers=headers)
    assert list_r.status_code == 200, list_r.text
    listing = list_r.json()
    assert listing["total"] >= 1
    job_ids = [item["job_id"] for item in listing["items"]]
    assert job_id in job_ids

    # Detail replays
    detail_r = await client.get(
        f"/api/v1/training/jobs/{job_id}", headers=headers
    )
    assert detail_r.status_code == 200, detail_r.text
    detail = detail_r.json()
    assert detail["registered_model_id"] == model_id

    # Models listing carries the model
    models_r = await client.get("/api/v1/models", headers=headers)
    assert models_r.status_code == 200, models_r.text
    models_body = models_r.json()
    assert models_body["total"] >= 1
    rows = {item["model_id"]: item for item in models_body["items"]}
    assert model_id in rows
    assert rows[model_id]["stage"] == "production", rows[model_id]

    # Detail
    md_r = await client.get(f"/api/v1/models/{model_id}", headers=headers)
    assert md_r.status_code == 200, md_r.text

    # Predict — the deadline-relevant passage should out-score the policy one.
    predict_r = await client.post(
        f"/api/v1/models/{model_id}/predict",
        headers=headers,
        json={
            "query": "When is the grant deadline?",
            "passages": [
                "The Modernized Public Records Disclosure Rule took effect "
                "on January 1, 2026.",
                "Applications must be submitted by February 28, 2026.",
                "Records officers should expect a 12-18% increase in volume.",
            ],
        },
    )
    assert predict_r.status_code == 200, predict_r.text
    pred = predict_r.json()
    assert pred["model_id"] == model_id
    assert len(pred["scored"]) == 3
    # Top result should be the grant-deadline passage (originally index 1).
    top = pred["scored"][0]
    assert top["index"] == 1, pred

    # Demote to archived
    promote_r = await client.post(
        f"/api/v1/models/{model_id}/promote",
        headers=headers,
        json={"stage": "archived", "notes": "decommissioned in test"},
    )
    assert promote_r.status_code == 200, promote_r.text
    assert promote_r.json()["stage"] == "archived"


@pytest.mark.asyncio
async def test_training_is_tenant_scoped(client) -> None:
    token_a = await _register(client, suffix="train-iso-a")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    r = await client.post(
        "/api/v1/training/jobs",
        headers=headers_a,
        json={
            "name": "psdi-cross-encoder-reranker",
            "auto_promote": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    job_id = body["job_id"]
    model_id = body["registered_model_id"]
    assert model_id, "should still register even without promotion"

    # Org B cannot read.
    token_b = await _register(client, suffix="train-iso-b")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    r1 = await client.get(
        f"/api/v1/training/jobs/{job_id}", headers=headers_b
    )
    assert r1.status_code == 404

    r2 = await client.get(f"/api/v1/models/{model_id}", headers=headers_b)
    assert r2.status_code == 404

    r3 = await client.post(
        f"/api/v1/models/{model_id}/predict",
        headers=headers_b,
        json={"query": "x", "passages": ["y"]},
    )
    assert r3.status_code == 404

    r4 = await client.post(
        f"/api/v1/models/{model_id}/promote",
        headers=headers_b,
        json={"stage": "archived"},
    )
    assert r4.status_code == 404


@pytest.mark.asyncio
async def test_training_rejects_unauthenticated(client) -> None:
    r = await client.post(
        "/api/v1/training/jobs", json={"auto_promote": False}
    )
    assert r.status_code == 401
