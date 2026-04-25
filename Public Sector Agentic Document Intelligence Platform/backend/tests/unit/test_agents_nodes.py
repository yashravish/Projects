"""Pure-unit tests for each agent node + the offline FakeLLM responder.

No DB, no LangGraph runtime; nodes are exercised as plain async functions.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from app.agents.llm_client import (
    ChatMessage,
    ChatResult,
    FakeLLM,
    TokenUsage,
    _offline_responder,  # type: ignore[attr-defined]
)
from app.agents.nodes import (
    CITATION_MARKER,
    _bind_citations,
    _safe_json_loads,
    critique_node,
    plan_node,
    retrieve_node,
    synthesize_node,
)
from app.agents.state import GraphState
from app.retrieval.hybrid import (
    RetrievalConfig,
    RetrievalSignal,
    RetrievedChunk,
)


def _chunk(idx: int, text: str, *, filename: str = "doc.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID(int=idx),
        document_id=uuid.UUID(int=99),
        document_filename=filename,
        organization_id=uuid.UUID(int=1),
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        chunk_index=idx,
        text=text,
        fused_score=1.0 / (idx + 1),
        bm25=RetrievalSignal(rank=idx + 1, score=0.5),
        vector=RetrievalSignal(rank=idx + 1, score=0.7),
    )


def _state(question: str = "What is the grant deadline?") -> GraphState:
    return {
        "organization_id": uuid.UUID(int=1),
        "user_id": None,
        "question": question,
        "model": "test-model",
        "trace": [],
        "token_usages": [],
        "error": None,
    }


# ---- helpers ------------------------------------------------------------------


def test_safe_json_loads_handles_code_fences() -> None:
    s = "```json\n{\"answer\": \"x\"}\n```"
    assert _safe_json_loads(s) == {"answer": "x"}


def test_safe_json_loads_returns_empty_on_garbage() -> None:
    assert _safe_json_loads("hello world") == {}


def test_bind_citations_drops_out_of_range_markers() -> None:
    chunks = [_chunk(0, "first"), _chunk(1, "second")]
    answer = "Claim one [1]. Claim two [3] (out of range)."
    cites = _bind_citations(answer, chunks)
    assert [c.index for c in cites] == [1]


def test_citation_marker_pattern() -> None:
    assert CITATION_MARKER.findall("[1] foo [12] bar [no]") == ["1", "12"]


# ---- offline FakeLLM responder ------------------------------------------------


def test_offline_responder_plan_passes_through_question() -> None:
    msg = [
        ChatMessage(role="system", content="..."),
        ChatMessage(role="user", content="# NODE: plan\n\nQUESTION:\nWhen does X close?\n"),
    ]
    out = json.loads(_offline_responder(msg))
    assert out == {"sub_questions": ["When does X close?"]}


def test_offline_responder_synthesize_quotes_first_evidence() -> None:
    msg = [
        ChatMessage(role="system", content="..."),
        ChatMessage(
            role="user",
            content=(
                "# NODE: synthesize\n\n"
                "QUESTION:\nWhat is X?\n\n"
                "EVIDENCE\n[1] (a.pdf, page 1-1)\nThe deadline is March 31, 2026.\n\n"
                "[2] (b.pdf, page 1-1)\nProposals are due in spring.\n"
            ),
        ),
    ]
    out = json.loads(_offline_responder(msg))
    assert "deadline is March 31" in out["answer"]
    assert "[1]" in out["answer"]
    assert out["used_indices"] == [1, 2]


def test_offline_responder_critique_returns_passing_envelope() -> None:
    msg = [ChatMessage(role="user", content="# NODE: critique\n\nANSWER:\n...")]
    out = json.loads(_offline_responder(msg))
    assert {"grounding_score", "hallucination_risk", "passed", "issues"} <= set(out)
    assert out["passed"] is True


# ---- node-level behaviour -----------------------------------------------------


@pytest.mark.asyncio
async def test_plan_node_falls_back_when_llm_returns_garbage() -> None:
    llm = FakeLLM(default_response="not json")
    out = await plan_node(_state("Q"), llm, model="m")
    assert out["sub_questions"] == ["Q"]
    assert out["trace"][-1].node == "plan"


@pytest.mark.asyncio
async def test_plan_node_caps_to_three_subquestions() -> None:
    llm = FakeLLM(
        default_response=json.dumps({"sub_questions": ["a", "b", "c", "d", "e"]})
    )
    out = await plan_node(_state(), llm, model="m")
    assert len(out["sub_questions"]) == 3


@pytest.mark.asyncio
async def test_retrieve_node_dedupes_across_subquestions() -> None:
    chunk_a = _chunk(0, "alpha alpha")
    chunk_b = _chunk(1, "beta beta")

    class _MockRetriever:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def retrieve(
            self,
            *,
            organization_id: uuid.UUID,
            query: str,
            config: RetrievalConfig | None = None,
        ) -> list[RetrievedChunk]:
            self.calls.append(query)
            # Both sub-questions surface chunk_a; only the first surfaces b.
            if "first" in query:
                return [chunk_a, chunk_b]
            return [chunk_a]

    state = _state()
    state["sub_questions"] = ["first sub-question", "second sub-question"]
    out = await retrieve_node(
        state,
        retriever=_MockRetriever(),  # type: ignore[arg-type]
        config=RetrievalConfig(top_k=5, candidate_k=5),
    )
    ids = {c.chunk_id for c in out["retrieved"]}
    assert ids == {chunk_a.chunk_id, chunk_b.chunk_id}


@pytest.mark.asyncio
async def test_synthesize_abstains_when_no_evidence() -> None:
    state = _state()
    state["retrieved"] = []
    llm = FakeLLM(default_response='{"answer": "should not be used"}')
    out = await synthesize_node(state, llm, model="m")
    assert "cannot answer" in out["answer_text"].lower()
    assert out["citations"] == []
    # Abstain path must NOT call the LLM.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_synthesize_binds_inline_citations_to_chunks() -> None:
    state = _state()
    state["retrieved"] = [_chunk(0, "First fact"), _chunk(1, "Second fact")]
    llm = FakeLLM(
        default_response=json.dumps(
            {"answer": "We learn one thing [1]. And another [2].", "used_indices": [1, 2]}
        )
    )
    out = await synthesize_node(state, llm, model="m")
    assert [c.index for c in out["citations"]] == [1, 2]
    assert out["citations"][0].chunk_id == uuid.UUID(int=0)
    assert out["citations"][1].chunk_id == uuid.UUID(int=1)


@pytest.mark.asyncio
async def test_critique_flags_unbound_marker_as_hallucination() -> None:
    state = _state()
    state["retrieved"] = [_chunk(0, "real thing")]
    state["answer_text"] = "Real claim [1]. Fabricated claim [9]."
    state["citations"] = _bind_citations(state["answer_text"], state["retrieved"])

    # FakeLLM cheerfully returns "passed=true"; the node must override it
    # because [9] does not bind to any chunk.
    llm = FakeLLM(
        default_response=json.dumps(
            {
                "grounding_score": 0.95,
                "hallucination_risk": 0.05,
                "passed": True,
                "issues": [],
            }
        )
    )
    out = await critique_node(state, llm, model="m")
    crit = out["critique"]
    assert crit.passed is False
    assert any("[9]" in i for i in crit.issues)
    assert crit.hallucination_risk >= 0.5


@pytest.mark.asyncio
async def test_critique_node_records_token_usage() -> None:
    state = _state()
    state["retrieved"] = [_chunk(0, "x")]
    state["answer_text"] = "Claim [1]."
    state["citations"] = _bind_citations(state["answer_text"], state["retrieved"])
    llm = FakeLLM(
        default_response=json.dumps(
            {
                "grounding_score": 0.9,
                "hallucination_risk": 0.1,
                "passed": True,
                "issues": [],
            }
        )
    )
    out = await critique_node(state, llm, model="m")
    usages = out["token_usages"]
    assert len(usages) == 1
    assert isinstance(usages[0], TokenUsage)
    assert usages[0].node == "critique"


# ---- typing sanity ------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_result_round_trip_keeps_node_tag() -> None:
    """Belt-and-braces: makes sure FakeLLM populates `node` correctly.

    NOTE: this test must NOT spin up its own event loop with `asyncio.run()`.
    Doing so closes pytest-asyncio's session loop and poisons every async
    test that runs after it (manifests as `coroutine was never awaited`).
    """
    llm = FakeLLM(default_response="{}")
    res: Any = await llm.chat(node="plan", model="m", messages=[])
    assert isinstance(res, ChatResult)
    assert res.usage.node == "plan"
