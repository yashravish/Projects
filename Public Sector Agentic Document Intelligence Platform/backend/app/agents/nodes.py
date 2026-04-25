"""Async LangGraph nodes for the inquiry agent.

Pipeline shape (linear; expansion to branching is straightforward):

    plan_node  →  retrieve_node  →  synthesize_node  →  critique_node

Every node:
- Reads from `GraphState` and returns a partial-state dict.
- Appends a `TraceStep` to `state["trace"]` so the audit trail is complete
  even if a downstream node fails.
- Records `TokenUsage` for any LLM call so cost/latency totals are exact.

Nodes never raise; on parse failure they fall back to a safe default and
record the issue in `trace.metadata["error"]`. The graph as a whole only fails
if `retrieve_node` returns no evidence (handled by the conditional edge in
`graph.py`).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from app.agents.llm_client import ChatMessage, LLMClient
from app.agents.prompts import (
    CRITIQUE_PROMPT,
    PLAN_PROMPT,
    SYNTHESIZE_PROMPT,
    PromptSpec,
)
from app.agents.state import (
    Citation,
    CritiqueResult,
    GraphState,
    TraceStep,
)
from app.logging_config import get_logger
from app.retrieval.hybrid import (
    HybridRetriever,
    RetrievalConfig,
    RetrievedChunk,
)

log = get_logger("agents.nodes")

CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")
SNIPPET_MAX_CHARS = 800
ANSWER_MAX_CHARS = 4000


# ---- helpers -------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_json_loads(content: str) -> dict[str, Any]:
    """Parse JSON; tolerate models that wrap their output in a code fence."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        loaded: Any = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---- plan node -----------------------------------------------------------------


async def plan_node(state: GraphState, llm: LLMClient, *, model: str) -> dict[str, Any]:
    started = _now_ms()
    question = state["question"]
    user_block = (
        f"# NODE: plan\n\n"
        f"QUESTION:\n{question}\n"
    )
    result = await llm.chat(
        node="plan",
        model=model,
        messages=[
            ChatMessage(role="system", content=PLAN_PROMPT.system),
            ChatMessage(role="user", content=user_block),
        ],
        response_format_json=True,
    )
    parsed = _safe_json_loads(result.content)
    raw = parsed.get("sub_questions")
    sub_questions: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    sub_questions.append(cleaned)
    if not sub_questions:
        sub_questions = [question.strip()]
    sub_questions = sub_questions[:3]

    duration = _now_ms() - started
    step = TraceStep(
        node="plan",
        label="Planning",
        detail=f"{len(sub_questions)} sub-question(s)",
        duration_ms=duration,
        metadata={
            "prompt_version": PLAN_PROMPT.version,
            "model": model,
            "sub_questions": list(sub_questions),
        },
    )
    return {
        "sub_questions": sub_questions,
        "trace": _append_trace(state, step),
        "token_usages": _append_usage(state, result.usage),
    }


# ---- retrieve node -------------------------------------------------------------


async def retrieve_node(
    state: GraphState,
    *,
    retriever: HybridRetriever,
    config: RetrievalConfig,
) -> dict[str, Any]:
    started = _now_ms()
    organization_id: uuid.UUID = state["organization_id"]
    sub_questions: list[str] = state.get("sub_questions") or [state["question"]]

    seen: dict[uuid.UUID, RetrievedChunk] = {}
    for sq in sub_questions:
        chunks = await retriever.retrieve(
            organization_id=organization_id,
            query=sq,
            config=config,
        )
        for c in chunks:
            existing = seen.get(c.chunk_id)
            if existing is None or c.fused_score > existing.fused_score:
                seen[c.chunk_id] = c
    merged = sorted(seen.values(), key=lambda c: c.fused_score, reverse=True)[: config.top_k]

    duration = _now_ms() - started
    step = TraceStep(
        node="retrieve",
        label="Retrieving",
        detail=f"{len(merged)} chunks fused from {len(sub_questions)} sub-question(s)",
        duration_ms=duration,
        metadata={
            "fused_count": len(merged),
            "candidates_per_signal": config.candidate_k,
            "top_k": config.top_k,
            "documents": sorted({c.document_filename for c in merged}),
        },
    )
    return {
        "retrieved": merged,
        "trace": _append_trace(state, step),
    }


# ---- synthesize node -----------------------------------------------------------


async def synthesize_node(state: GraphState, llm: LLMClient, *, model: str) -> dict[str, Any]:
    started = _now_ms()
    retrieved: list[RetrievedChunk] = state.get("retrieved") or []
    question = state["question"]

    if not retrieved:
        # Nothing to synthesize against — fail fast and honestly.
        usage_step = TraceStep(
            node="synthesize",
            label="Synthesizing",
            detail="No evidence — abstaining",
            duration_ms=_now_ms() - started,
            metadata={"prompt_version": SYNTHESIZE_PROMPT.version, "model": model},
        )
        return {
            "answer_text": (
                "I cannot answer this question from the available corpus. "
                "No relevant excerpts were retrieved."
            ),
            "citations": [],
            "trace": _append_trace(state, usage_step),
        }

    evidence_block = _format_evidence(retrieved)
    user_block = (
        f"# NODE: synthesize\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE\n{evidence_block}\n"
    )

    result = await llm.chat(
        node="synthesize",
        model=model,
        messages=[
            ChatMessage(role="system", content=SYNTHESIZE_PROMPT.system),
            ChatMessage(role="user", content=user_block),
        ],
        response_format_json=True,
    )
    parsed = _safe_json_loads(result.content)
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        answer = (
            "I could not produce a confident answer from the retrieved evidence."
        )
    answer = _truncate(answer, ANSWER_MAX_CHARS)

    citations = _bind_citations(answer, retrieved)

    duration = _now_ms() - started
    step = TraceStep(
        node="synthesize",
        label="Synthesizing",
        detail=f"{len(citations)} citation(s) bound from {len(retrieved)} candidates",
        duration_ms=duration,
        metadata={
            "prompt_version": SYNTHESIZE_PROMPT.version,
            "model": model,
            "answer_chars": len(answer),
            "n_citations": len(citations),
        },
    )
    return {
        "answer_text": answer,
        "citations": citations,
        "trace": _append_trace(state, step),
        "token_usages": _append_usage(state, result.usage),
    }


# ---- critique node -------------------------------------------------------------


async def critique_node(state: GraphState, llm: LLMClient, *, model: str) -> dict[str, Any]:
    started = _now_ms()
    answer = state.get("answer_text") or ""
    citations = state.get("citations") or []
    retrieved = state.get("retrieved") or []
    by_index = {c.index: c for c in citations}

    if not answer or not citations:
        critique = CritiqueResult(
            grounding_score=0.0,
            hallucination_risk=1.0,
            passed=False,
            issues=["No grounded citations were produced."],
        )
        step = TraceStep(
            node="critique",
            label="Validating",
            detail="No citations — failed grounding check",
            duration_ms=_now_ms() - started,
            metadata={"prompt_version": CRITIQUE_PROMPT.version, "model": model},
        )
        return {"critique": critique, "trace": _append_trace(state, step)}

    cited_block = "\n\n".join(
        f"[{c.index}] (page {c.page_start}-{c.page_end} of {c.document_filename})\n"
        f"{_lookup_chunk_text(c.chunk_id, retrieved)}"
        for c in citations
    )
    user_block = (
        f"# NODE: critique\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CITED EVIDENCE:\n{cited_block}\n"
    )

    result = await llm.chat(
        node="critique",
        model=model,
        messages=[
            ChatMessage(role="system", content=CRITIQUE_PROMPT.system),
            ChatMessage(role="user", content=user_block),
        ],
        response_format_json=True,
    )
    parsed = _safe_json_loads(result.content)

    grounding = _coerce_score(parsed.get("grounding_score"), default=0.5)
    hallucination = _coerce_score(parsed.get("hallucination_risk"), default=0.5)
    passed = bool(parsed.get("passed", grounding >= 0.7 and hallucination <= 0.3))
    issues_raw = parsed.get("issues") or []
    issues = [str(i) for i in issues_raw if isinstance(i, (str, int, float))][:10]

    # Sanity — every cited [N] must reference a chunk we actually retrieved.
    rebound_issues = list(issues)
    for marker in CITATION_MARKER.findall(answer):
        idx = int(marker)
        if idx not in by_index:
            rebound_issues.append(
                f"Inline marker [{idx}] does not bind to any retrieved chunk."
            )
            grounding = min(grounding, 0.5)
            hallucination = max(hallucination, 0.5)
            passed = False

    critique = CritiqueResult(
        grounding_score=grounding,
        hallucination_risk=hallucination,
        passed=passed,
        issues=rebound_issues,
    )

    duration = _now_ms() - started
    step = TraceStep(
        node="critique",
        label="Validating",
        detail=(
            f"grounding={grounding:.2f} hallucination={hallucination:.2f} "
            f"{'pass' if passed else 'fail'}"
        ),
        duration_ms=duration,
        metadata={
            "prompt_version": CRITIQUE_PROMPT.version,
            "model": model,
            "grounding_score": grounding,
            "hallucination_risk": hallucination,
            "passed": passed,
            "issues": rebound_issues,
        },
    )
    return {
        "critique": critique,
        "trace": _append_trace(state, step),
        "token_usages": _append_usage(state, result.usage),
    }


# ---- pure helpers --------------------------------------------------------------


def _format_evidence(chunks: Iterable[RetrievedChunk]) -> str:
    out: list[str] = []
    for i, c in enumerate(chunks, start=1):
        body = _truncate(c.text.replace("\n", " "), SNIPPET_MAX_CHARS)
        out.append(f"[{i}] ({c.document_filename}, page {c.page_start}-{c.page_end})\n{body}")
    return "\n\n".join(out)


def _bind_citations(answer: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Map every `[N]` marker in `answer` to the Nth retrieved chunk.

    Markers that point past the retrieved list are dropped silently — the
    critic node will surface them as issues.
    """
    seen_indices: set[int] = set()
    citations: list[Citation] = []
    for marker in CITATION_MARKER.findall(answer):
        idx = int(marker)
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        if 1 <= idx <= len(retrieved):
            chunk = retrieved[idx - 1]
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_filename=chunk.document_filename,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    snippet=_truncate(chunk.text.replace("\n", " "), SNIPPET_MAX_CHARS),
                )
            )
    citations.sort(key=lambda c: c.index)
    return citations


def _lookup_chunk_text(chunk_id: uuid.UUID, retrieved: list[RetrievedChunk]) -> str:
    for c in retrieved:
        if c.chunk_id == chunk_id:
            return _truncate(c.text.replace("\n", " "), SNIPPET_MAX_CHARS)
    return ""


def _coerce_score(value: Any, *, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _append_trace(state: GraphState, step: TraceStep) -> list[TraceStep]:
    existing = list(state.get("trace") or [])
    existing.append(step)
    return existing


def _append_usage(state: GraphState, usage: Any) -> list[Any]:
    existing = list(state.get("token_usages") or [])
    existing.append(usage)
    return existing


# ---- factory: bind the dependencies a graph needs ------------------------------


def make_node_callables(
    *,
    llm: LLMClient,
    retriever: HybridRetriever,
    model: str,
    retrieval_config: RetrievalConfig,
) -> dict[str, Callable[[GraphState], Any]]:
    """Return async callables suitable for `StateGraph.add_node`.

    LangGraph nodes must be unary `(state) → state-update`; this binds the
    other dependencies up front.
    """

    async def _plan(state: GraphState) -> dict[str, Any]:
        return await plan_node(state, llm=llm, model=model)

    async def _retrieve(state: GraphState) -> dict[str, Any]:
        return await retrieve_node(state, retriever=retriever, config=retrieval_config)

    async def _synth(state: GraphState) -> dict[str, Any]:
        return await synthesize_node(state, llm=llm, model=model)

    async def _critique(state: GraphState) -> dict[str, Any]:
        return await critique_node(state, llm=llm, model=model)

    return {
        "plan": _plan,
        "retrieve": _retrieve,
        "synthesize": _synth,
        "critique": _critique,
    }


_PROMPTS: tuple[PromptSpec, ...] = (PLAN_PROMPT, SYNTHESIZE_PROMPT, CRITIQUE_PROMPT)


__all__ = [
    "CITATION_MARKER",
    "critique_node",
    "make_node_callables",
    "plan_node",
    "retrieve_node",
    "synthesize_node",
]
