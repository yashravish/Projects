"""Pure-function metric computations for the evaluation harness.

Every score is a float in [0, 1] (or 0/1 boolean cast to float) so they
aggregate cleanly. No metric here mutates the input or hits the DB.

Conventions:
  * Higher is better for *most* metrics. The two exceptions
    (`hallucination_risk`, `forbidden_phrase_rate`) are documented at their
    callsites and emitted alongside the others as-is — the aggregate-pass
    rule below takes their direction into account.
  * "Pass" is a single composite verdict combining critic + retrieval +
    faithfulness so the leaderboard can show one column without flattening.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Iterable, Sequence

from app.agents.state import InquiryResult
from app.eval.dataset import GoldItem


@dataclasses.dataclass(frozen=True)
class ItemMetrics:
    """All scores for one (gold_item, inquiry_result) pair."""

    item_id: str
    retrieval_recall: float
    retrieval_precision: float
    citation_precision: float
    citation_recall: float
    faithfulness: float
    forbidden_phrase_rate: float
    grounding_score: float
    hallucination_risk: float
    answer_passed_critic: bool
    item_passed: bool
    latency_ms: int
    n_retrieved: int
    n_citations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "retrieval_recall": round(self.retrieval_recall, 4),
            "retrieval_precision": round(self.retrieval_precision, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "forbidden_phrase_rate": round(self.forbidden_phrase_rate, 4),
            "grounding_score": round(self.grounding_score, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "answer_passed_critic": self.answer_passed_critic,
            "item_passed": self.item_passed,
            "latency_ms": self.latency_ms,
            "n_retrieved": self.n_retrieved,
            "n_citations": self.n_citations,
        }


def score_item(*, gold: GoldItem, result: InquiryResult) -> ItemMetrics:
    """Compute every metric for one inquiry result against its gold item."""
    expected = {f.lower() for f in gold.expected_doc_filenames}

    retrieved_docs = [c.document_filename.lower() for c in result.retrieved]
    cited_docs = [c.document_filename.lower() for c in result.citations]

    recall = _set_recall(expected, retrieved_docs)
    precision = _set_precision(expected, retrieved_docs)
    cite_precision = (
        _set_precision(expected, cited_docs) if cited_docs else 0.0
    )
    cite_recall = _set_recall(expected, cited_docs)

    faithfulness = _faithfulness(result.answer_text, gold.must_contain_any)
    forbidden_rate = _forbidden_phrase_rate(
        result.answer_text, gold.forbidden_phrases
    )

    answer_passed = bool(result.critique.passed)
    # Composite item-pass: enough retrieval + enough faithfulness + critic pass
    # + no forbidden phrases. Tunable thresholds, but conservative defaults.
    item_passed = (
        recall >= 0.5
        and faithfulness >= 0.5
        and forbidden_rate == 0.0
        and answer_passed
    )

    return ItemMetrics(
        item_id=gold.id,
        retrieval_recall=recall,
        retrieval_precision=precision,
        citation_precision=cite_precision,
        citation_recall=cite_recall,
        faithfulness=faithfulness,
        forbidden_phrase_rate=forbidden_rate,
        grounding_score=float(result.critique.grounding_score),
        hallucination_risk=float(result.critique.hallucination_risk),
        answer_passed_critic=answer_passed,
        item_passed=item_passed,
        latency_ms=int(result.total_latency_ms),
        n_retrieved=len(result.retrieved),
        n_citations=len(result.citations),
    )


@dataclasses.dataclass(frozen=True)
class AggregateMetrics:
    """Mean / median / pass-rate roll-up across a dataset."""

    n_items: int
    pass_rate: float
    retrieval_recall: float
    retrieval_precision: float
    citation_precision: float
    citation_recall: float
    faithfulness: float
    forbidden_phrase_rate: float
    grounding_score: float
    hallucination_risk: float
    latency_ms_p50: float
    latency_ms_p95: float
    n_failures: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "pass_rate": round(self.pass_rate, 4),
            "retrieval_recall": round(self.retrieval_recall, 4),
            "retrieval_precision": round(self.retrieval_precision, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "forbidden_phrase_rate": round(self.forbidden_phrase_rate, 4),
            "grounding_score": round(self.grounding_score, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "latency_ms_p50": round(self.latency_ms_p50, 2),
            "latency_ms_p95": round(self.latency_ms_p95, 2),
            "n_failures": self.n_failures,
        }


def aggregate(items: Sequence[ItemMetrics]) -> AggregateMetrics:
    """Roll a list of `ItemMetrics` into a single `AggregateMetrics`."""
    if not items:
        return AggregateMetrics(
            n_items=0,
            pass_rate=0.0,
            retrieval_recall=0.0,
            retrieval_precision=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            faithfulness=0.0,
            forbidden_phrase_rate=0.0,
            grounding_score=0.0,
            hallucination_risk=0.0,
            latency_ms_p50=0.0,
            latency_ms_p95=0.0,
            n_failures=0,
        )

    n = len(items)
    latencies = sorted(it.latency_ms for it in items)
    return AggregateMetrics(
        n_items=n,
        pass_rate=sum(1 for it in items if it.item_passed) / n,
        retrieval_recall=_mean(it.retrieval_recall for it in items),
        retrieval_precision=_mean(it.retrieval_precision for it in items),
        citation_precision=_mean(it.citation_precision for it in items),
        citation_recall=_mean(it.citation_recall for it in items),
        faithfulness=_mean(it.faithfulness for it in items),
        forbidden_phrase_rate=_mean(it.forbidden_phrase_rate for it in items),
        grounding_score=_mean(it.grounding_score for it in items),
        hallucination_risk=_mean(it.hallucination_risk for it in items),
        latency_ms_p50=_percentile(latencies, 50),
        latency_ms_p95=_percentile(latencies, 95),
        n_failures=sum(1 for it in items if not it.item_passed),
    )


def _set_recall(expected: set[str], observed: Iterable[str]) -> float:
    if not expected:
        return 1.0
    obs = set(observed)
    return len(expected & obs) / len(expected)


def _set_precision(expected: set[str], observed: Sequence[str]) -> float:
    if not observed:
        return 0.0
    return sum(1 for o in observed if o in expected) / len(observed)


def _faithfulness(
    answer_text: str, must_contain_any: tuple[tuple[str, ...], ...]
) -> float:
    """Fraction of OR-groups whose phrases appear (case-insensitive) in answer."""
    if not must_contain_any:
        return 1.0
    haystack = answer_text.lower()
    hits = 0
    for group in must_contain_any:
        if any(p.lower() in haystack for p in group):
            hits += 1
    return hits / len(must_contain_any)


def _forbidden_phrase_rate(
    answer_text: str, forbidden: tuple[str, ...]
) -> float:
    """Fraction of forbidden phrases that appear. 0.0 is good."""
    if not forbidden:
        return 0.0
    haystack = answer_text.lower()
    hits = sum(1 for p in forbidden if p.lower() in haystack)
    return hits / len(forbidden)


def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return float(statistics.fmean(vs)) if vs else 0.0


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default behavior)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


__all__ = [
    "AggregateMetrics",
    "ItemMetrics",
    "aggregate",
    "score_item",
]
