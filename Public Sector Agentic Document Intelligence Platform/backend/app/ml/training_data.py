"""Deterministic synthesis of (query, passage, label) triples for the
cross-encoder reranker.

The platform is shipped with three seeded PDFs and a versioned gold-question
dataset (see `app.eval.dataset`). We use those two sources together to mint a
small but well-shaped training set with three classes of rows:

  * **Positive**  — the query *and* a passage from the expected document
                    that contains at least one phrase from the matching
                    `must_contain_any` group. Labelled `1`.

  * **Hard negative** — the query and a passage from the *expected*
                        document that does *not* contain any of the gold
                        phrases. (Same topic, wrong span.) Labelled `0`.

  * **Easy negative** — the query and a passage from a *different* document
                        on a different topic. Labelled `0`.

Determinism: passage chunking, candidate selection, and shuffling all use a
fixed seed and stable iteration over `GOLD_DATASET.items`. The output is
content-addressed via a SHA256 of the canonical JSONL serialisation; the
training script writes this hash to the manifest so a registered model is
traceable to the exact data that produced it.

The corpus is not loaded from the live Postgres database. We re-render the
sample PDFs in-process via `app.seed.generate_sample_pdfs.build_sample_pdfs`
and chunk their text with `app.ingestion.chunker`, so this module is usable
in CI, in a SageMaker training container, and from a local Python REPL — all
without any DB or network dependency.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
from typing import Iterable, Iterator, Sequence

from app.chunking.chunker import ChunkConfig, chunk_pages
from app.chunking.pdf_extractor import extract_pdf
from app.eval.dataset import GOLD_DATASET, GoldItem, GoldQuestionDataset
from app.seed.generate_sample_pdfs import SamplePDF, build_sample_pdfs

# We intentionally pick chunk sizes well below the production retriever's so
# the training rows are fine-grained enough that a "right span" really is
# distinguishable from a "wrong span" in the same document.
_TRAIN_CHUNK_CFG = ChunkConfig(
    target_chars=600,
    max_chars=900,
    min_chars=120,
    overlap_chars=80,
)

_SEED = 0xC0_DE_FE_ED


@dataclasses.dataclass(frozen=True)
class TrainingExample:
    """One labelled (query, passage) row for cross-encoder training."""

    query: str
    passage: str
    label: int  # 1 = relevant, 0 = irrelevant
    source_document: str  # filename of the source PDF
    gold_item_id: str  # e.g. "grant-deadline" — the originating gold case
    kind: str  # "positive" | "hard_negative" | "easy_negative"

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "passage": self.passage,
            "label": self.label,
            "source_document": self.source_document,
            "gold_item_id": self.gold_item_id,
            "kind": self.kind,
        }


@dataclasses.dataclass(frozen=True)
class TrainingTriples:
    """A bundle of training rows + a content-addressed identity."""

    rows: tuple[TrainingExample, ...]
    dataset_name: str
    """Name of the upstream gold dataset (lineage)."""
    dataset_version: str
    """Version hash of the upstream gold dataset (lineage)."""

    @property
    def fingerprint(self) -> str:
        """Stable short hash of the training rows for lineage in the manifest."""
        payload = json.dumps(
            {
                "dataset_name": self.dataset_name,
                "dataset_version": self.dataset_version,
                "rows": [r.as_dict() for r in self.rows],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.rows)

    def label_counts(self) -> dict[int, int]:
        out: dict[int, int] = {0: 0, 1: 0}
        for r in self.rows:
            out[r.label] = out.get(r.label, 0) + 1
        return out

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    def to_jsonl(self) -> str:
        """Canonical JSONL serialisation — one row per line, sorted keys."""
        return "\n".join(
            json.dumps(r.as_dict(), sort_keys=True, ensure_ascii=False)
            for r in self.rows
        )

    @classmethod
    def from_jsonl(cls, *, text: str, dataset_name: str, dataset_version: str) -> "TrainingTriples":
        rows: list[TrainingExample] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                TrainingExample(
                    query=str(d["query"]),
                    passage=str(d["passage"]),
                    label=int(d["label"]),
                    source_document=str(d["source_document"]),
                    gold_item_id=str(d.get("gold_item_id") or ""),
                    kind=str(d.get("kind") or "unknown"),
                )
            )
        return cls(
            rows=tuple(rows),
            dataset_name=dataset_name,
            dataset_version=dataset_version,
        )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Whitespace-collapse for fair phrase matching."""
    return re.sub(r"\s+", " ", text).strip()


def _phrase_in_text(phrase: str, text: str) -> bool:
    return phrase.lower() in text.lower()


def _passage_satisfies_groups(
    passage: str, groups: Sequence[Sequence[str]]
) -> bool:
    """A passage is "positive" if it contains at least one phrase from EACH
    `must_contain_any` group — same rule the evaluator uses."""
    if not groups:
        return False
    for group in groups:
        if not any(_phrase_in_text(p, passage) for p in group):
            return False
    return True


def _passage_contains_any(passage: str, phrases: Iterable[str]) -> bool:
    return any(_phrase_in_text(p, passage) for p in phrases)


def _chunk_corpus(
    pdfs: Sequence[SamplePDF],
) -> dict[str, list[str]]:
    """Return {filename -> list of passage texts}. Deterministic order."""
    out: dict[str, list[str]] = {}
    for pdf in pdfs:
        pages = extract_pdf(pdf.bytes_)
        page_tuples = [(p.page_number, p.text) for p in pages]
        chunks = chunk_pages(page_tuples, _TRAIN_CHUNK_CFG)
        out[pdf.filename] = [_normalise(c.text) for c in chunks]
    return out


def _select_positive(
    *,
    passages: Sequence[str],
    item: GoldItem,
) -> str | None:
    """Return the first passage that satisfies ALL `must_contain_any` groups,
    falling back to one that hits at least the first group, then None."""
    for p in passages:
        if _passage_satisfies_groups(p, item.must_contain_any):
            return p
    if item.must_contain_any:
        first_group = item.must_contain_any[0]
        for p in passages:
            if _passage_contains_any(p, first_group):
                return p
    return None


def _select_hard_negative(
    *,
    passages: Sequence[str],
    item: GoldItem,
    rng: random.Random,
) -> str | None:
    """A "wrong span" from the *same* document — rich same-domain noise."""
    candidates = [
        p
        for p in passages
        if not _passage_satisfies_groups(p, item.must_contain_any)
        and not _passage_contains_any(p, item.forbidden_phrases or ())
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _select_easy_negative(
    *,
    corpus: dict[str, list[str]],
    item: GoldItem,
    rng: random.Random,
) -> tuple[str, str] | None:
    """A passage from a different document entirely — easy negative."""
    others = [
        (fname, passages)
        for fname, passages in corpus.items()
        if fname not in item.expected_doc_filenames and passages
    ]
    if not others:
        return None
    fname, passages = rng.choice(others)
    return fname, rng.choice(passages)


# ── Public API ───────────────────────────────────────────────────────────────


def synthesize_training_triples(
    *,
    dataset: GoldQuestionDataset | None = None,
    pdfs: Sequence[SamplePDF] | None = None,
    seed: int = _SEED,
) -> TrainingTriples:
    """Build the training rows.

    Per gold item, mints one positive (when one is findable in the document),
    one hard negative (different span in the same document), and two easy
    negatives (from other documents). The result is a small (~30–50 row)
    balanced classification dataset that's plenty for a TF-IDF + LR
    cross-encoder to learn from.

    Determinism is total: same gold dataset + same PDFs + same seed → same
    rows in the same order with the same fingerprint.
    """
    ds = dataset if dataset is not None else GOLD_DATASET
    pdfs = pdfs if pdfs is not None else build_sample_pdfs()

    corpus = _chunk_corpus(pdfs)
    rng = random.Random(seed)
    rows: list[TrainingExample] = []

    for item in ds.items:
        # Pick the document(s) the item targets. We treat the FIRST expected
        # filename as canonical for choosing positive + hard-negative passages
        # (the gold set never lists more than one for a given item today).
        target_fname = item.expected_doc_filenames[0]
        target_passages = corpus.get(target_fname, [])
        if not target_passages:
            continue

        positive = _select_positive(passages=target_passages, item=item)
        if positive is not None:
            rows.append(
                TrainingExample(
                    query=item.question,
                    passage=positive,
                    label=1,
                    source_document=target_fname,
                    gold_item_id=item.id,
                    kind="positive",
                )
            )

        hard_neg = _select_hard_negative(
            passages=target_passages, item=item, rng=rng
        )
        if hard_neg is not None:
            rows.append(
                TrainingExample(
                    query=item.question,
                    passage=hard_neg,
                    label=0,
                    source_document=target_fname,
                    gold_item_id=item.id,
                    kind="hard_negative",
                )
            )

        # Two easy negatives per item, drawn without replacement of source
        # filename when possible so we get topical breadth.
        seen_easy: set[str] = set()
        attempts = 0
        while len(seen_easy) < 2 and attempts < 6:
            attempts += 1
            picked = _select_easy_negative(corpus=corpus, item=item, rng=rng)
            if picked is None:
                break
            other_fname, other_passage = picked
            if other_fname in seen_easy:
                continue
            seen_easy.add(other_fname)
            rows.append(
                TrainingExample(
                    query=item.question,
                    passage=other_passage,
                    label=0,
                    source_document=other_fname,
                    gold_item_id=item.id,
                    kind="easy_negative",
                )
            )

    # Stable shuffle so labels are interleaved, not blocked.
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    rows = [r for _, r in indexed]

    return TrainingTriples(
        rows=tuple(rows),
        dataset_name=ds.name,
        dataset_version=ds.version,
    )


__all__ = [
    "TrainingExample",
    "TrainingTriples",
    "synthesize_training_triples",
]
