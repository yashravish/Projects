"""LocalDeterministicEmbedder behavior."""
from __future__ import annotations

import math

import pytest

from app.agents.llm_client import EMBEDDING_DIM, LocalDeterministicEmbedder


@pytest.mark.asyncio
async def test_returns_unit_vectors_of_correct_dim() -> None:
    emb = LocalDeterministicEmbedder()
    vecs = await emb.embed(["alpha beta", "gamma delta"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == EMBEDDING_DIM
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, rel=1e-5)


@pytest.mark.asyncio
async def test_deterministic_across_calls() -> None:
    emb = LocalDeterministicEmbedder()
    a = (await emb.embed(["resilient communities grant"]))[0]
    b = (await emb.embed(["resilient communities grant"]))[0]
    assert a == b


@pytest.mark.asyncio
async def test_distinguishes_distinct_texts() -> None:
    emb = LocalDeterministicEmbedder()
    a, b = await emb.embed(["procurement notice", "policy memo public records"])
    # Cosine similarity well below 1.0 — i.e. these aren't the same vector.
    sim = sum(x * y for x, y in zip(a, b, strict=True))
    assert sim < 0.95


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list() -> None:
    emb = LocalDeterministicEmbedder()
    assert await emb.embed([]) == []
