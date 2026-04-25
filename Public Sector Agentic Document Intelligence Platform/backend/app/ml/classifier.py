"""Cross-encoder reranker classifier.

The model is a deliberately small, deterministic, dependency-light pipeline
that can be trained in seconds inside a CI runner or a SageMaker training
container without a GPU and without any external model download:

    Pipeline:
      1. Concatenate query and passage with a sentinel separator.
      2. Char-and-word TF-IDF feature extraction.
      3. Logistic regression with class-weight balancing.

The output is a probability in [0, 1] that "passage is relevant to query".
At inference time the platform's `Reranker` calls `score_pairs` and uses the
returned scores to re-order the top-K hybrid retrieval candidates.

Why not a real cross-encoder transformer? Two reasons:
  1. The whole pipeline must be runnable on first `docker compose up` with
     no network — that rules out anything that downloads weights at boot.
  2. Stage 5 is about exercising the *training/registry/serving loop* end to
     end. A small well-shaped sklearn model is enough to demonstrate every
     concern (lineage, metrics, promotion, latency) and the same loop will
     accept a fine-tuned transformer the moment the team needs one — only
     `CrossEncoderClassifier` would need replacing.

Persistence: a single `joblib` artifact (`model.joblib`) plus a sidecar
`manifest.json`. The manifest holds versioning + lineage and is what the
registry indexes on.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from app.ml.training_data import TrainingExample

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"
TRAINING_DATA_FILENAME = "training_data.jsonl"

QP_SEPARATOR = " ⟂ "
"""Sentinel between query and passage in the encoded text. Picked so it
will not appear in any natural document and is preserved verbatim by the
default TF-IDF token pattern."""


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    """All knobs that affect training output. Persisted in the manifest."""

    word_max_features: int = 20_000
    word_ngram_max: int = 2
    char_max_features: int = 20_000
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    logreg_C: float = 1.0
    logreg_max_iter: int = 500
    holdout_fraction: float = 0.25
    seed: int = 0xC0_DE_FE_ED

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TrainingMetrics:
    """Quality metrics computed on the training + holdout split."""

    n_train: int
    n_holdout: int
    train_accuracy: float
    holdout_accuracy: float
    holdout_precision: float
    holdout_recall: float
    holdout_f1: float
    holdout_roc_auc: float
    holdout_avg_precision: float
    holdout_log_loss: float
    score_separation: float
    """Mean(score | label=1) − Mean(score | label=0) on the holdout split.
    A friendly summary metric a practitioner can read at a glance: > 0.4
    means the model is meaningfully ranking positives above negatives."""

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def _encode_pair(query: str, passage: str) -> str:
    return f"{query.strip()}{QP_SEPARATOR}{passage.strip()}"


def _make_pipeline(hp: Hyperparameters) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, hp.word_ngram_max),
                                lowercase=True,
                                max_features=hp.word_max_features,
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(hp.char_ngram_min, hp.char_ngram_max),
                                lowercase=True,
                                max_features=hp.char_max_features,
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=hp.logreg_C,
                    max_iter=hp.logreg_max_iter,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=hp.seed,
                ),
            ),
        ]
    )


def _stratified_split(
    examples: Sequence[TrainingExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Per-label deterministic split. Sized to behave even at low n."""
    by_label: dict[int, list[TrainingExample]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    rng = np.random.default_rng(seed)
    train: list[TrainingExample] = []
    holdout: list[TrainingExample] = []
    for label, rows in sorted(by_label.items()):
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        n_holdout = max(1, int(round(len(rows) * fraction))) if len(rows) >= 2 else 0
        holdout.extend(rows[i] for i in idx[:n_holdout])
        train.extend(rows[i] for i in idx[n_holdout:])
    return train, holdout


class CrossEncoderClassifier:
    """A tiny, fast, deterministic relevance classifier.

    The classifier is trained against a `Sequence[TrainingExample]` and
    persisted as a single joblib artifact.
    """

    def __init__(
        self, *, hyperparameters: Hyperparameters | None = None
    ) -> None:
        self._hp = hyperparameters or Hyperparameters()
        self._pipeline: Pipeline | None = None

    @property
    def hyperparameters(self) -> Hyperparameters:
        return self._hp

    def fit(
        self, examples: Sequence[TrainingExample]
    ) -> TrainingMetrics:
        """Fit on `examples` and return holdout metrics.

        Uses a label-stratified holdout split sized by `holdout_fraction`.
        On very small datasets (< 8 rows or only one class) the holdout
        metrics fall back to a "all-train" report so the API still returns
        a complete metrics record without crashing.
        """
        if not examples:
            raise ValueError("cannot train: examples is empty")
        labels = {ex.label for ex in examples}
        if labels != {0, 1}:
            raise ValueError(
                f"training set must contain both labels {{0, 1}}; got {labels!r}"
            )

        train, holdout = _stratified_split(
            examples, fraction=self._hp.holdout_fraction, seed=self._hp.seed
        )
        if not train:
            train = list(examples)
            holdout = []

        # Edge case: stratified split gave a single-class train fold (e.g. 2
        # rows where the holdout took the only positive). Roll back to all-train.
        if len({ex.label for ex in train}) < 2:
            train = list(examples)
            holdout = []

        X_train = [_encode_pair(ex.query, ex.passage) for ex in train]
        y_train = np.array([ex.label for ex in train], dtype=np.int64)

        pipeline = _make_pipeline(self._hp)
        pipeline.fit(X_train, y_train)
        self._pipeline = pipeline

        train_pred = pipeline.predict(X_train)
        train_accuracy = float(accuracy_score(y_train, train_pred))

        if holdout and len({ex.label for ex in holdout}) > 1:
            X_holdout = [_encode_pair(ex.query, ex.passage) for ex in holdout]
            y_holdout = np.array([ex.label for ex in holdout], dtype=np.int64)
            scores = pipeline.predict_proba(X_holdout)[:, 1]
            preds = (scores >= 0.5).astype(np.int64)
            metrics = TrainingMetrics(
                n_train=len(train),
                n_holdout=len(holdout),
                train_accuracy=train_accuracy,
                holdout_accuracy=float(accuracy_score(y_holdout, preds)),
                holdout_precision=float(precision_score(y_holdout, preds, zero_division=0)),
                holdout_recall=float(recall_score(y_holdout, preds, zero_division=0)),
                holdout_f1=float(f1_score(y_holdout, preds, zero_division=0)),
                holdout_roc_auc=float(roc_auc_score(y_holdout, scores)),
                holdout_avg_precision=float(average_precision_score(y_holdout, scores)),
                holdout_log_loss=float(
                    log_loss(y_holdout, scores, labels=[0, 1])
                ),
                score_separation=float(
                    np.mean(scores[y_holdout == 1])
                    - np.mean(scores[y_holdout == 0])
                ),
            )
        else:
            # All-train report: honest about the absence of a holdout.
            scores_train = pipeline.predict_proba(X_train)[:, 1]
            metrics = TrainingMetrics(
                n_train=len(train),
                n_holdout=0,
                train_accuracy=train_accuracy,
                holdout_accuracy=train_accuracy,
                holdout_precision=float(
                    precision_score(y_train, train_pred, zero_division=0)
                ),
                holdout_recall=float(
                    recall_score(y_train, train_pred, zero_division=0)
                ),
                holdout_f1=float(f1_score(y_train, train_pred, zero_division=0)),
                holdout_roc_auc=float(
                    roc_auc_score(y_train, scores_train)
                    if len(set(y_train.tolist())) > 1
                    else 0.5
                ),
                holdout_avg_precision=float(
                    average_precision_score(y_train, scores_train)
                    if len(set(y_train.tolist())) > 1
                    else 0.5
                ),
                holdout_log_loss=float(
                    log_loss(y_train, scores_train, labels=[0, 1])
                ),
                score_separation=float(
                    np.mean(scores_train[y_train == 1])
                    - np.mean(scores_train[y_train == 0])
                ),
            )
        return metrics

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return P(relevant=1) for each (query, passage) pair."""
        if self._pipeline is None:
            raise RuntimeError("classifier is not fitted")
        if not pairs:
            return []
        encoded = [_encode_pair(q, p) for q, p in pairs]
        probs = self._pipeline.predict_proba(encoded)[:, 1]
        return [float(x) for x in probs]

    def is_fitted(self) -> bool:
        return self._pipeline is not None

    # ---- persistence ------------------------------------------------------

    def save(
        self,
        directory: str | pathlib.Path,
        *,
        manifest: dict[str, Any],
        metrics: TrainingMetrics,
    ) -> dict[str, str]:
        """Persist the fitted model + manifest + metrics. Returns the file
        layout `{role: absolute_path}` for downstream registry indexing."""
        if self._pipeline is None:
            raise RuntimeError("cannot save: classifier is not fitted")

        d = pathlib.Path(directory).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)

        model_path = d / MODEL_FILENAME
        joblib.dump(self._pipeline, model_path)

        manifest_path = d / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        metrics_path = d / METRICS_FILENAME
        metrics_path.write_text(
            json.dumps(metrics.as_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

        return {
            "model": str(model_path),
            "manifest": str(manifest_path),
            "metrics": str(metrics_path),
            "directory": str(d),
        }


def load_classifier(
    directory: str | pathlib.Path,
) -> CrossEncoderClassifier:
    """Inverse of `CrossEncoderClassifier.save`."""
    d = pathlib.Path(directory).expanduser().resolve()
    model_path = d / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"no fitted model at {model_path!s}")
    pipeline = joblib.load(model_path)

    classifier = CrossEncoderClassifier()
    classifier._pipeline = pipeline
    return classifier


__all__ = [
    "CrossEncoderClassifier",
    "Hyperparameters",
    "MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "MODEL_FILENAME",
    "QP_SEPARATOR",
    "TRAINING_DATA_FILENAME",
    "TrainingMetrics",
    "load_classifier",
]
