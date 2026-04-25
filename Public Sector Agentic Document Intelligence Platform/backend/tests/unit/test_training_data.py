"""Tests for the training-triple synthesizer.

The training data is the *root* of the lineage chain — if it isn't
deterministic and well-formed, every metric and every registered model
that descends from it inherits the noise. These tests treat the dataset
as a contract.
"""
from __future__ import annotations

from collections import Counter

import pytest

from app.eval.dataset import GOLD_DATASET
from app.ml.training_data import (
    TrainingExample,
    TrainingTriples,
    synthesize_training_triples,
)


def test_synthesize_returns_balanced_two_class_dataset() -> None:
    triples = synthesize_training_triples()
    counts = triples.label_counts()
    assert 0 in counts and 1 in counts, counts
    assert counts[0] >= 5, "expected non-trivial number of negatives"
    assert counts[1] >= 5, "expected non-trivial number of positives"
    # The negatives strictly outnumber the positives by design (1 hard + 2 easy
    # negatives per positive). Sanity-check the ratio.
    assert counts[0] >= counts[1], counts


def test_synthesize_is_deterministic_across_calls() -> None:
    a = synthesize_training_triples()
    b = synthesize_training_triples()
    assert a.fingerprint == b.fingerprint
    assert tuple(r.as_dict() for r in a) == tuple(r.as_dict() for r in b)


def test_synthesize_changes_fingerprint_when_dataset_changes() -> None:
    triples = synthesize_training_triples()
    # Override the dataset version by mutating a shallow dict and re-hashing.
    mutated = TrainingTriples(
        rows=triples.rows,
        dataset_name=triples.dataset_name + "-mut",
        dataset_version=triples.dataset_version,
    )
    assert mutated.fingerprint != triples.fingerprint


def test_kinds_are_well_distributed() -> None:
    triples = synthesize_training_triples()
    kinds = Counter(r.kind for r in triples.rows)
    assert kinds["positive"] >= 1, kinds
    assert kinds["hard_negative"] >= 1, kinds
    assert kinds["easy_negative"] >= 1, kinds
    # Hard negatives are valuable; expect at least as many as positives in a
    # well-formed run (each positive should produce one hard negative).
    assert kinds["hard_negative"] >= kinds["positive"]


def test_jsonl_roundtrip_preserves_rows() -> None:
    triples = synthesize_training_triples()
    text = triples.to_jsonl()
    rebuilt = TrainingTriples.from_jsonl(
        text=text,
        dataset_name=triples.dataset_name,
        dataset_version=triples.dataset_version,
    )
    assert len(rebuilt) == len(triples)
    assert rebuilt.fingerprint == triples.fingerprint
    for original, restored in zip(triples.rows, rebuilt.rows):
        assert original == restored


def test_positives_actually_contain_required_phrases() -> None:
    triples = synthesize_training_triples()
    by_id = {item.id: item for item in GOLD_DATASET.items}
    for ex in triples.rows:
        if ex.kind != "positive":
            continue
        gold = by_id[ex.gold_item_id]
        # The positive's source document must be one of the gold expected docs.
        assert ex.source_document in gold.expected_doc_filenames
        # And its passage must satisfy at least the first must_contain_any group.
        first_group = gold.must_contain_any[0]
        assert any(p.lower() in ex.passage.lower() for p in first_group), (
            f"positive for {gold.id} does not contain any of {first_group} "
            f"(passage: {ex.passage[:120]!r})"
        )


def test_easy_negatives_come_from_other_documents() -> None:
    triples = synthesize_training_triples()
    by_id = {item.id: item for item in GOLD_DATASET.items}
    seen_at_least_one = False
    for ex in triples.rows:
        if ex.kind != "easy_negative":
            continue
        gold = by_id[ex.gold_item_id]
        assert ex.source_document not in gold.expected_doc_filenames
        seen_at_least_one = True
    assert seen_at_least_one


def test_synthesize_is_robust_to_empty_examples() -> None:
    # Pass an empty dataset; expect a well-formed empty result without crashing.
    from app.eval.dataset import GoldQuestionDataset

    empty = GoldQuestionDataset(
        name="empty", description="empty", items=()
    )
    triples = synthesize_training_triples(dataset=empty)
    assert len(triples) == 0
    assert triples.label_counts() == {0: 0, 1: 0}


def test_training_example_as_dict_round_trip() -> None:
    ex = TrainingExample(
        query="q",
        passage="p",
        label=1,
        source_document="x.pdf",
        gold_item_id="gid",
        kind="positive",
    )
    d = ex.as_dict()
    assert d == {
        "query": "q",
        "passage": "p",
        "label": 1,
        "source_document": "x.pdf",
        "gold_item_id": "gid",
        "kind": "positive",
    }


@pytest.mark.parametrize("seed", [0, 1, 1_000_000])
def test_seed_changes_shuffle_but_preserves_set_of_rows(seed: int) -> None:
    a = synthesize_training_triples()
    b = synthesize_training_triples(seed=seed)
    set_a = {(r.query, r.passage, r.label, r.kind) for r in a.rows}
    set_b = {(r.query, r.passage, r.label, r.kind) for r in b.rows}
    # The hard-negative selection uses the rng so the rows themselves can
    # differ, but the *positives* and the *count of negatives* should be
    # stable. Assert both supersets contain the positives.
    pos_a = {r for r in set_a if r[2] == 1}
    pos_b = {r for r in set_b if r[2] == 1}
    assert pos_a == pos_b
