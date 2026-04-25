"""Tests for the cross-encoder classifier.

We don't assert against arbitrary metric thresholds because the dataset is
small and noisy; instead we assert *structural* invariants of training
(it learns *something* — positives score higher than negatives) and the
persist→load round-trip preserves predictions exactly.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    CrossEncoderClassifier,
    Hyperparameters,
    TrainingMetrics,
    load_classifier,
)
from app.ml.training_data import (
    TrainingExample,
    synthesize_training_triples,
)


def _toy_examples() -> list[TrainingExample]:
    """Tiny but realistic training set for unit-level tests."""
    return [
        TrainingExample(
            query="When is the grant deadline?",
            passage="Applications must be submitted by February 28, 2026.",
            label=1,
            source_document="grant.pdf",
            gold_item_id="grant-deadline",
            kind="positive",
        ),
        TrainingExample(
            query="When is the grant deadline?",
            passage="The procurement vendor portal is available at portal.gov.",
            label=0,
            source_document="procurement.pdf",
            gold_item_id="grant-deadline",
            kind="easy_negative",
        ),
        TrainingExample(
            query="What is the program ceiling?",
            passage="The total program ceiling for fiscal year 2026 is $740,000,000.",
            label=1,
            source_document="grant.pdf",
            gold_item_id="grant-ceiling",
            kind="positive",
        ),
        TrainingExample(
            query="What is the program ceiling?",
            passage="Records officers should expect a 12-18% increase in volume.",
            label=0,
            source_document="policy.pdf",
            gold_item_id="grant-ceiling",
            kind="easy_negative",
        ),
        TrainingExample(
            query="When are proposals due for procurement 117?",
            passage="Proposals are due by 4:00 PM local time on March 18, 2026.",
            label=1,
            source_document="procurement.pdf",
            gold_item_id="procurement-due-date",
            kind="positive",
        ),
        TrainingExample(
            query="When are proposals due for procurement 117?",
            passage="The Modernized Public Records Disclosure Rule took effect.",
            label=0,
            source_document="policy.pdf",
            gold_item_id="procurement-due-date",
            kind="easy_negative",
        ),
        # An extra hard negative pair so stratification has room to split.
        TrainingExample(
            query="What is the program ceiling?",
            passage="Eligible applicants are units of state, local, tribal, or territorial government.",
            label=0,
            source_document="grant.pdf",
            gold_item_id="grant-ceiling",
            kind="hard_negative",
        ),
        TrainingExample(
            query="When is the grant deadline?",
            passage="Performance shall begin no later than April 15, 2026.",
            label=0,
            source_document="procurement.pdf",
            gold_item_id="grant-deadline",
            kind="easy_negative",
        ),
    ]


def test_fit_runs_and_returns_well_formed_metrics() -> None:
    clf = CrossEncoderClassifier()
    metrics = clf.fit(_toy_examples())
    assert isinstance(metrics, TrainingMetrics)
    assert metrics.n_train >= 1
    assert 0.0 <= metrics.holdout_accuracy <= 1.0
    assert 0.0 <= metrics.train_accuracy <= 1.0
    assert clf.is_fitted()


def test_classifier_learns_to_separate_positives_from_negatives() -> None:
    clf = CrossEncoderClassifier()
    clf.fit(_toy_examples())
    pos_score = clf.score_pairs(
        [
            (
                "When is the grant deadline?",
                "Applications must be submitted by February 28, 2026.",
            )
        ]
    )[0]
    neg_score = clf.score_pairs(
        [
            (
                "When is the grant deadline?",
                "Quarterly performance reports are required during the period.",
            )
        ]
    )[0]
    assert pos_score > neg_score, (pos_score, neg_score)


def test_score_pairs_requires_fitted_pipeline() -> None:
    clf = CrossEncoderClassifier()
    with pytest.raises(RuntimeError):
        clf.score_pairs([("q", "p")])


def test_fit_rejects_single_class() -> None:
    examples = [
        TrainingExample(
            query="q",
            passage="p",
            label=1,
            source_document="x",
            gold_item_id="g",
            kind="positive",
        ),
        TrainingExample(
            query="q2",
            passage="p2",
            label=1,
            source_document="x",
            gold_item_id="g",
            kind="positive",
        ),
    ]
    with pytest.raises(ValueError):
        CrossEncoderClassifier().fit(examples)


def test_save_and_load_round_trip(tmp_path: pathlib.Path) -> None:
    clf = CrossEncoderClassifier()
    metrics = clf.fit(_toy_examples())
    artifacts = clf.save(
        tmp_path,
        manifest={"name": "x", "version": "v1", "framework": "sklearn-tfidf-logreg"},
        metrics=metrics,
    )
    assert (tmp_path / MODEL_FILENAME).is_file()
    assert (tmp_path / MANIFEST_FILENAME).is_file()
    assert (tmp_path / METRICS_FILENAME).is_file()
    assert artifacts["model"] == str(tmp_path / MODEL_FILENAME)

    loaded = load_classifier(tmp_path)
    pairs = [
        (
            "When are proposals due for procurement 117?",
            "Proposals are due by 4:00 PM local time on March 18, 2026.",
        ),
        ("When is the grant deadline?", "Records officers should expect a 12-18% increase."),
    ]
    assert clf.score_pairs(pairs) == loaded.score_pairs(pairs)

    # Manifest content is JSON-decodable + has the expected name.
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["name"] == "x"
    assert manifest["version"] == "v1"


def test_load_classifier_raises_on_missing_artifact(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "does-not-exist")


def test_hyperparameters_round_trip() -> None:
    hp = Hyperparameters(logreg_C=2.0, char_ngram_max=4)
    d = hp.as_dict()
    assert d["logreg_C"] == 2.0
    assert d["char_ngram_max"] == 4


def test_classifier_handles_synth_dataset_end_to_end(tmp_path: pathlib.Path) -> None:
    """Smoke: the real synthesised dataset trains without crashing."""
    triples = synthesize_training_triples()
    clf = CrossEncoderClassifier()
    metrics = clf.fit(list(triples))
    artifacts = clf.save(
        tmp_path,
        manifest={"name": "smoke", "version": "v1"},
        metrics=metrics,
    )
    assert artifacts["model"]
    # Score the seeded positive of `procurement-due-date` higher than a
    # random non-relevant passage from the policy doc.
    pos_score = clf.score_pairs(
        [
            (
                "When are proposals due for Procurement Notice 2026-117?",
                "Proposals are due by 4:00 PM local time on March 18, 2026.",
            )
        ]
    )[0]
    neg_score = clf.score_pairs(
        [
            (
                "When are proposals due for Procurement Notice 2026-117?",
                "The Modernized Public Records Disclosure Rule took effect "
                "on January 1, 2026.",
            )
        ]
    )[0]
    assert pos_score > neg_score
