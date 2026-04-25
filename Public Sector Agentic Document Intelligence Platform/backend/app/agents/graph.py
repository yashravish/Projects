"""Compose the four nodes into a LangGraph `StateGraph` and a runner facade.

The graph is intentionally linear:

    [START] → plan → retrieve → synthesize → critique → [END]

`plan` is the only node that can shrink the work — when it returns a single
sub-question identical to the original, the rest of the graph behaves as if
no decomposition happened.

We use LangGraph for the executor-level guarantees (typed channels, tracing
hooks, checkpointing-ready) but keep the node bodies in plain async Python
functions in `nodes.py` so they remain unit-testable without LangGraph
dependencies in the test path.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.llm_client import LLMClient
from app.agents.nodes import make_node_callables
from app.agents.state import (
    Citation,
    CritiqueResult,
    GraphState,
    InquiryResult,
    TraceStep,
)
from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.hybrid import (
    HybridRetriever,
    RetrievalConfig,
    RetrievedChunk,
)

log = get_logger("agents.graph")


def build_inquiry_graph(
    *,
    llm: LLMClient,
    retriever: HybridRetriever,
    model: str,
    retrieval_config: RetrievalConfig,
) -> Any:
    """Return a compiled LangGraph for the inquiry pipeline.

    The compiled object exposes `.ainvoke(state) -> state`.
    """
    callables = make_node_callables(
        llm=llm,
        retriever=retriever,
        model=model,
        retrieval_config=retrieval_config,
    )
    # NOTE: LangGraph forbids node names that collide with `GraphState` keys
    # (it would create an ambiguous channel). `critique` is a state key, so the
    # validating node is named `validate` here. The trace step still records
    # `node="critique"` so the audit trail remains stable.
    graph: StateGraph = StateGraph(GraphState)
    graph.add_node("plan_node", callables["plan"])
    graph.add_node("retrieve_node", callables["retrieve"])
    graph.add_node("synthesize_node", callables["synthesize"])
    graph.add_node("validate_node", callables["critique"])

    graph.add_edge(START, "plan_node")
    graph.add_edge("plan_node", "retrieve_node")
    graph.add_edge("retrieve_node", "synthesize_node")
    graph.add_edge("synthesize_node", "validate_node")
    graph.add_edge("validate_node", END)

    return graph.compile()


class GraphRunner:
    """Stateless facade that runs one inquiry to completion.

    Initialize once per request — `HybridRetriever` holds an `AsyncSession`
    that must not outlive the request.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        retriever: HybridRetriever,
        retrieval_config: RetrievalConfig | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._config = retrieval_config or RetrievalConfig()
        self._model = model or get_settings().openai_default_model
        self._compiled = build_inquiry_graph(
            llm=llm,
            retriever=retriever,
            model=self._model,
            retrieval_config=self._config,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def retrieval_config(self) -> RetrievalConfig:
        return self._config

    async def run(
        self,
        *,
        organization_id: uuid.UUID,
        user_id: uuid.UUID | None,
        question: str,
    ) -> InquiryResult:
        if not question.strip():
            raise ValueError("question must be non-empty")

        initial: GraphState = {
            "organization_id": organization_id,
            "user_id": user_id,
            "question": question.strip(),
            "model": self._model,
            "trace": [],
            "token_usages": [],
            "error": None,
        }
        wall_start = time.monotonic()
        try:
            final: dict[str, Any] = await self._compiled.ainvoke(initial)
        except Exception as exc:  # noqa: BLE001 — graph fail must not 500 silently
            log.exception(
                "graph.failed",
                organization_id=str(organization_id),
                question_preview=question[:80],
            )
            duration_ms = int((time.monotonic() - wall_start) * 1000)
            return InquiryResult(
                question=question.strip(),
                answer_text="The inquiry pipeline failed before producing an answer.",
                citations=[],
                retrieved=[],
                critique=CritiqueResult(
                    grounding_score=0.0,
                    hallucination_risk=1.0,
                    passed=False,
                    issues=[f"pipeline_error: {type(exc).__name__}"],
                ),
                trace=[
                    TraceStep(
                        node="error",
                        label="Pipeline failure",
                        detail=str(exc)[:200],
                        duration_ms=duration_ms,
                        metadata={"exception": type(exc).__name__},
                    )
                ],
                token_usages=[],
                model=self._model,
                error=f"{type(exc).__name__}: {exc}",
            )

        retrieved: list[RetrievedChunk] = list(final.get("retrieved") or [])
        citations: list[Citation] = list(final.get("citations") or [])
        critique: CritiqueResult | None = final.get("critique")
        if critique is None:
            critique = CritiqueResult(
                grounding_score=0.0,
                hallucination_risk=1.0,
                passed=False,
                issues=["critique node did not produce output"],
            )
        return InquiryResult(
            question=question.strip(),
            answer_text=str(final.get("answer_text") or "").strip(),
            citations=citations,
            retrieved=retrieved,
            critique=critique,
            trace=list(final.get("trace") or []),
            token_usages=list(final.get("token_usages") or []),
            model=self._model,
            error=None,
        )


__all__ = ["GraphRunner", "build_inquiry_graph"]
