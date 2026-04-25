"""Machine learning subsystem for the platform.

Trains and serves a cross-encoder reranker that rescores the top-K candidates
returned by the hybrid retriever. Designed to run identically on a local
host (subprocess + filesystem registry) and on AWS SageMaker (training job +
Model Package Group), behind a single backend abstraction.

Public surface:
    * `training_data` — deterministic synthesis of (query, passage, label)
      rows against the seeded corpus + gold-question dataset.
    * `classifier`    — the sklearn cross-encoder pipeline (train/predict/save/load).
    * `training_script` — the SageMaker-compatible entry-point script.
    * `backends`       — `TrainingBackend` protocol + Local/SageMaker impls.
    * `registry`       — `ModelRegistry` protocol + Local/SageMaker impls.
    * `reranker`       — `Reranker` protocol + Local/SageMaker/Null impls
                          (consumed by the retrieval layer at inference time).
    * `factory`        — picks the right impls from `Settings`.
"""
from __future__ import annotations

from app.ml.classifier import (
    CrossEncoderClassifier,
    Hyperparameters,
    TrainingMetrics,
    load_classifier,
)
from app.ml.training_data import (
    TrainingExample,
    TrainingTriples,
    synthesize_training_triples,
)

__all__ = [
    "CrossEncoderClassifier",
    "Hyperparameters",
    "TrainingExample",
    "TrainingMetrics",
    "TrainingTriples",
    "load_classifier",
    "synthesize_training_triples",
]
