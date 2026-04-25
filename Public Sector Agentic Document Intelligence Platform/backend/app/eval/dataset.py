"""The gold-question regression dataset.

This is the authoritative truth set the evaluation harness scores against.
It is *committed to source* and *versioned by content hash* so that:

  * Every `EvaluationRun` row knows exactly which dataset version produced it.
  * A change to any field (question text, expected docs, required phrases)
    forces a new `dataset_version`, so a re-run is not silently comparable
    to a prior run.

Each item targets one of the three seeded PDFs (see
`app/seed/generate_sample_pdfs.py`). The expected_doc_filenames must match
those filenames *exactly* — the metrics module compares as sets.

`must_contain_any` lets us tolerate paraphrase ("February 28" vs
"February 28, 2026") while still pinning that the answer must mention the
canonical fact in *some* form. `forbidden_phrases` catches the most likely
hallucinations the offline FakeLLM, or any model, might emit.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Sequence


@dataclasses.dataclass(frozen=True)
class GoldItem:
    """One gold-question regression case."""

    id: str
    question: str
    expected_doc_filenames: tuple[str, ...]
    must_contain_any: tuple[tuple[str, ...], ...]
    """Tuple of OR-groups. The answer is credited if at least one phrase in
    each group appears (case-insensitive)."""
    forbidden_phrases: tuple[str, ...] = ()
    topic: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_doc_filenames": list(self.expected_doc_filenames),
            "must_contain_any": [list(g) for g in self.must_contain_any],
            "forbidden_phrases": list(self.forbidden_phrases),
            "topic": self.topic,
        }


@dataclasses.dataclass(frozen=True)
class GoldQuestionDataset:
    """A named, content-addressed bundle of gold items."""

    name: str
    description: str
    items: tuple[GoldItem, ...]

    @property
    def version(self) -> str:
        """Stable short hash over the canonical JSON serialisation.

        Sorting keys recursively makes the hash insensitive to dict-key order
        across Python versions, but sensitive to any actual content change.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "items": [it.as_dict() for it in self.items],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[GoldItem]:
        return iter(self.items)


# ────────────────────────────────────────────────────────────────────────────
# The default gold dataset.
#
# Rule of thumb: every fact asserted here must be literally present in the
# corresponding PDF (see `generate_sample_pdfs.py`) so that an offline,
# evidence-quoting agent has a fair shot. If you change the PDFs, update this
# dataset and the version hash will rotate.
# ────────────────────────────────────────────────────────────────────────────


GOLD_ITEMS: Sequence[GoldItem] = (
    # --- Grant program ----------------------------------------------------
    GoldItem(
        id="grant-deadline",
        question="What is the application deadline for the Resilient Communities Infrastructure Grant?",
        expected_doc_filenames=("fy26-resilient-communities-grant.pdf",),
        must_contain_any=(
            ("February 28, 2026", "February 28", "Feb. 28, 2026", "Feb 28"),
            ("11:59 PM", "11:59 pm", "11:59"),
        ),
        forbidden_phrases=("March 1, 2026", "January 28, 2026"),
        topic="grants",
    ),
    GoldItem(
        id="grant-ceiling",
        question="What is the FY26 program ceiling for the Resilient Communities grant, and what is the cost-share for jurisdictions over 25,000 population?",
        expected_doc_filenames=("fy26-resilient-communities-grant.pdf",),
        must_contain_any=(
            ("$740,000,000", "740,000,000", "$740 million"),
            ("25%",),
        ),
        forbidden_phrases=("$1,000,000,000", "$10,000,000,000"),
        topic="grants",
    ),
    GoldItem(
        id="grant-disaster-zone",
        question="How does the Resilient Communities grant define a 'disaster zone'?",
        expected_doc_filenames=("fy26-resilient-communities-grant.pdf",),
        must_contain_any=(
            ("36 months",),
            ("federally declared", "major disaster", "42 U.S.C."),
        ),
        topic="grants",
    ),
    GoldItem(
        id="grant-scoring",
        question="What is the minimum score required to advance to the programmatic review stage in the Resilient Communities grant?",
        expected_doc_filenames=("fy26-resilient-communities-grant.pdf",),
        must_contain_any=(
            ("70",),
            ("programmatic review",),
        ),
        topic="grants",
    ),
    # --- Procurement ------------------------------------------------------
    GoldItem(
        id="procurement-due-date",
        question="When are proposals due for Procurement Notice 2026-117?",
        expected_doc_filenames=("procurement-notice-2026-117.pdf",),
        must_contain_any=(
            ("March 18, 2026", "Mar 18, 2026", "Mar. 18", "March 18"),
            ("4:00 PM", "4 pm", "4:00", "4 p.m."),
        ),
        forbidden_phrases=("March 19, 2026", "April 18"),
        topic="procurement",
    ),
    GoldItem(
        id="procurement-estimated-value",
        question="What is the estimated contract value range for the pavement condition assessment procurement?",
        expected_doc_filenames=("procurement-notice-2026-117.pdf",),
        must_contain_any=(
            ("$1.8M", "1.8 million", "1,800,000"),
            ("$2.4M", "2.4 million", "2,400,000"),
        ),
        topic="procurement",
    ),
    GoldItem(
        id="procurement-set-aside",
        question="What set-aside category and NAICS code apply to Procurement Notice 2026-117?",
        expected_doc_filenames=("procurement-notice-2026-117.pdf",),
        must_contain_any=(
            ("small business", "small-business"),
            ("541370",),
        ),
        topic="procurement",
    ),
    # --- Policy memo -------------------------------------------------------
    GoldItem(
        id="policy-response-window",
        question="What is the standard response window under the Modernized Public Records Disclosure Rule, and what extension is allowed?",
        expected_doc_filenames=("policy-memo-public-records.pdf",),
        must_contain_any=(
            ("20-business-day", "20 business day", "20 business-day"),
            ("10-business-day", "10 business day", "10 business-day"),
        ),
        forbidden_phrases=("30-business-day", "60 calendar days"),
        topic="policy",
    ),
    GoldItem(
        id="policy-effective-date",
        question="When did the Modernized Public Records Disclosure Rule take effect?",
        expected_doc_filenames=("policy-memo-public-records.pdf",),
        must_contain_any=(
            ("January 1, 2026", "Jan. 1, 2026", "Jan 1, 2026"),
        ),
        forbidden_phrases=("January 1, 2025", "March 1, 2026"),
        topic="policy",
    ),
    GoldItem(
        id="policy-quarterly-report",
        question="When is the first quarterly transparency report under the new public records rule due?",
        expected_doc_filenames=("policy-memo-public-records.pdf",),
        must_contain_any=(
            ("May 15, 2026", "May 15"),
        ),
        topic="policy",
    ),
)


GOLD_DATASET = GoldQuestionDataset(
    name="publicsector-adip-gold-v1",
    description=(
        "Gold-question regression set covering all three seeded "
        "PublicSector ADIP corpus documents. Each item asserts a verifiable "
        "fact present verbatim in the source PDF."
    ),
    items=tuple(GOLD_ITEMS),
)


_REGISTRY: dict[str, GoldQuestionDataset] = {GOLD_DATASET.name: GOLD_DATASET}


def get_dataset(name: str | None = None) -> GoldQuestionDataset:
    """Return a registered dataset by name, or the default if `name` is None."""
    if not name:
        return GOLD_DATASET
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown evaluation dataset {name!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "GOLD_DATASET",
    "GoldItem",
    "GoldQuestionDataset",
    "get_dataset",
]
