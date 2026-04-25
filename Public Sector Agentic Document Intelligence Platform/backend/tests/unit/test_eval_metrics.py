"""Pure-function tests for `app.eval.metrics`.

These tests construct synthetic `InquiryResult` records (no DB, no graph)
so they pin the metric semantics independently of the agent.
"""
from __future__ import annotations

import uuid

from app.agents.state import Citation, CritiqueResult, InquiryResult, TraceStep
from app.eval.dataset import GoldItem
from app.eval.metrics import aggregate, score_item
from app.retrieval.hybrid import RetrievalSignal, RetrievedChunk


def _chunk(filename: str, *, doc_id: uuid.UUID | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        document_filename=filename,
        organization_id=uuid.uuid4(),
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
        chunk_index=0,
        text="canonical text",
        bm25=RetrievalSignal(rank=0, score=1.0),
        vector=RetrievalSignal(rank=0, score=1.0),
        fused_score=0.5,
    )


def _citation(filename: str, index: int = 1) -> Citation:
    return Citation(
        index=index,
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_filename=filename,
        page_start=1,
        page_end=1,
        snippet="snippet",
    )


def _result(
    *,
    answer: str,
    retrieved_filenames: list[str],
    cited_filenames: list[str],
    grounding: float = 0.9,
    halluc: float = 0.1,
    passed: bool = True,
    duration_ms: int = 100,
) -> InquiryResult:
    return InquiryResult(
        question="q",
        answer_text=answer,
        citations=[_citation(f, i + 1) for i, f in enumerate(cited_filenames)],
        retrieved=[_chunk(f) for f in retrieved_filenames],
        critique=CritiqueResult(
            grounding_score=grounding,
            hallucination_risk=halluc,
            passed=passed,
            issues=[],
        ),
        trace=[TraceStep("plan", "Plan", "ok", duration_ms)],
        token_usages=[],
        model="fake",
        error=None,
    )


GOLD = GoldItem(
    id="g1",
    question="q",
    expected_doc_filenames=("doc-a.pdf",),
    must_contain_any=(("hello",), ("world",)),
    forbidden_phrases=("definitely-wrong",),
    topic="t",
)


def test_perfect_answer_passes_all_metrics() -> None:
    res = _result(
        answer="hello world: the answer",
        retrieved_filenames=["doc-a.pdf", "doc-b.pdf"],
        cited_filenames=["doc-a.pdf"],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.retrieval_recall == 1.0
    assert m.retrieval_precision == 0.5  # 1 of 2 retrieved is expected
    assert m.citation_recall == 1.0
    assert m.citation_precision == 1.0
    assert m.faithfulness == 1.0
    assert m.forbidden_phrase_rate == 0.0
    assert m.item_passed is True


def test_missing_required_phrase_fails_faithfulness() -> None:
    # Only one of the two OR-groups satisfied.
    res = _result(
        answer="hello: but the other word is missing",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.faithfulness == 0.5
    # 0.5 is the inclusive threshold for item_passed; exactly 0.5 → still passes.
    assert m.item_passed is True


def test_below_faithfulness_threshold_fails_item() -> None:
    res = _result(
        answer="empty answer",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.faithfulness == 0.0
    assert m.item_passed is False


def test_forbidden_phrase_blows_up_pass() -> None:
    res = _result(
        answer="hello world but also definitely-wrong",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.faithfulness == 1.0
    assert m.forbidden_phrase_rate == 1.0
    assert m.item_passed is False, "forbidden phrase must veto pass"


def test_retrieval_recall_zero_blocks_pass() -> None:
    res = _result(
        answer="hello world",
        retrieved_filenames=["other.pdf"],
        cited_filenames=[],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.retrieval_recall == 0.0
    assert m.item_passed is False


def test_critic_failure_blocks_pass() -> None:
    res = _result(
        answer="hello world",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
        passed=False,
        grounding=0.3,
        halluc=0.6,
    )
    m = score_item(gold=GOLD, result=res)
    assert m.answer_passed_critic is False
    assert m.item_passed is False


def test_no_citations_does_not_crash() -> None:
    res = _result(
        answer="hello world",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=[],
    )
    m = score_item(gold=GOLD, result=res)
    assert m.citation_recall == 0.0
    assert m.citation_precision == 0.0  # no citations → defined as 0 not NaN


def test_aggregate_pass_rate_and_percentiles() -> None:
    res_ok = _result(
        answer="hello world",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
        duration_ms=100,
    )
    res_bad = _result(
        answer="empty",
        retrieved_filenames=["doc-a.pdf"],
        cited_filenames=["doc-a.pdf"],
        duration_ms=900,
    )
    m1 = score_item(gold=GOLD, result=res_ok)
    m2 = score_item(gold=GOLD, result=res_bad)
    agg = aggregate([m1, m2])
    assert agg.n_items == 2
    assert agg.pass_rate == 0.5
    assert agg.n_failures == 1
    assert agg.latency_ms_p50 == 500.0  # midpoint of 100 and 900
    assert agg.latency_ms_p95 >= 500.0


def test_aggregate_empty_is_zeroed() -> None:
    agg = aggregate([])
    assert agg.n_items == 0
    assert agg.pass_rate == 0.0
    assert agg.latency_ms_p50 == 0.0
