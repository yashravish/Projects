"""Inference-time reranker.

Three implementations behind one Protocol:

  * `LocalReranker` — loads a `joblib`-persisted classifier from disk and
                       scores (query, passage) pairs in-process. The
                       classifier is cached (LRU) so concurrent requests
                       don't re-load it.

  * `SageMakerReranker` — invokes a SageMaker realtime endpoint via boto3.
                           Body is `{"query": ..., "passages": [...]}` →
                           response `{"scores": [...]}`.

  * `NullReranker` — pass-through; preserves input order. Used as the
                      fallback when no production model is registered yet
                      so the rest of the pipeline never has to special-case
                      the empty registry.

The retrieval layer fuses BM25 + vector signals first (RRF), then *if* a
reranker is selected and a production model is in service, the top
`candidate_k` are re-scored and re-sorted before the top `top_k` are
returned. The fused-score is preserved on the chunk for audit trail; the
new score is stored as `rerank_score` so a downstream consumer can tell why
the order changed.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Protocol, cast, runtime_checkable

from app.config import Settings, get_settings
from app.ml.boto_typing import Boto3Module, SageMakerRuntimeClient
from app.logging_config import get_logger
from app.ml.classifier import CrossEncoderClassifier, load_classifier

log = get_logger("ml.reranker")


@dataclasses.dataclass(frozen=True)
class RerankRequest:
    query: str
    passages: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RerankResult:
    scores: tuple[float, ...]
    """Same length and order as `passages`; higher = more relevant."""
    backend: str
    model_name: str
    model_version: str


@runtime_checkable
class Reranker(Protocol):
    """The contract every reranker must satisfy."""

    backend: str
    model_name: str
    model_version: str

    def score(
        self, request: RerankRequest
    ) -> RerankResult:  # pragma: no cover — Protocol declaration
        ...


# ── Local reranker ───────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=8)
def _cached_classifier(local_dir: str) -> CrossEncoderClassifier:
    """LRU-cache classifiers by their on-disk directory.

    A registry update with the same directory will *not* invalidate this —
    the registry guarantees per-version directories are write-once.
    """
    log.info("ml.reranker.local.load", local_dir=local_dir)
    return load_classifier(local_dir)


@dataclasses.dataclass
class LocalReranker:
    local_dir: str
    model_name: str
    model_version: str
    backend: str = "local"

    def score(self, request: RerankRequest) -> RerankResult:
        if not request.passages:
            return RerankResult(
                scores=(),
                backend=self.backend,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        clf = _cached_classifier(self.local_dir)
        pairs = [(request.query, p) for p in request.passages]
        scores = clf.score_pairs(pairs)
        return RerankResult(
            scores=tuple(scores),
            backend=self.backend,
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ── SageMaker reranker ───────────────────────────────────────────────────────


@dataclasses.dataclass
class SageMakerReranker:
    settings: Settings
    endpoint_name: str
    model_name: str
    model_version: str
    backend: str = "sagemaker"
    boto3_module: Boto3Module | None = None

    def _client(self) -> SageMakerRuntimeClient:
        boto3: Boto3Module
        if self.boto3_module is not None:
            boto3 = self.boto3_module
        else:
            import boto3 as _boto3

            boto3 = _boto3
        return cast(
            SageMakerRuntimeClient,
            boto3.client(
                "sagemaker-runtime", region_name=self.settings.aws_region
            ),
        )

    def score(self, request: RerankRequest) -> RerankResult:
        if not request.passages:
            return RerankResult(
                scores=(),
                backend=self.backend,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        import json

        body = json.dumps(
            {"query": request.query, "passages": list(request.passages)}
        ).encode("utf-8")
        response = self._client().invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        payload = response["Body"].read()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        decoded = json.loads(payload)
        scores = decoded.get("scores")
        if not isinstance(scores, list) or len(scores) != len(request.passages):
            raise RuntimeError(
                "SageMaker reranker returned malformed scores; "
                f"expected list of length {len(request.passages)}, got {decoded!r}"
            )
        return RerankResult(
            scores=tuple(float(s) for s in scores),
            backend=self.backend,
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ── Null reranker ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class NullReranker:
    """No-op reranker — preserves input order via uniform scores."""

    backend: str = "null"
    model_name: str = "null"
    model_version: str = "0"

    def score(self, request: RerankRequest) -> RerankResult:
        return RerankResult(
            scores=tuple(0.5 for _ in request.passages),
            backend=self.backend,
            model_name=self.model_name,
            model_version=self.model_version,
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def build_reranker_from_handle(
    *,
    name: str,
    version: str,
    local_dir: str,
    settings: Settings | None = None,
) -> Reranker:
    """Build the right reranker for a registered artifact."""
    s = settings or get_settings()
    if (
        s.reranker_backend == "sagemaker"
        and s.sagemaker_reranker_endpoint
    ):
        return SageMakerReranker(
            settings=s,
            endpoint_name=s.sagemaker_reranker_endpoint,
            model_name=name,
            model_version=version,
        )
    return LocalReranker(
        local_dir=local_dir,
        model_name=name,
        model_version=version,
    )


def reset_reranker_cache_for_tests() -> None:
    """Drop the in-process classifier cache. Tests should call this when
    they swap a model artifact between assertions."""
    _cached_classifier.cache_clear()


__all__ = [
    "LocalReranker",
    "NullReranker",
    "RerankRequest",
    "RerankResult",
    "Reranker",
    "SageMakerReranker",
    "build_reranker_from_handle",
    "reset_reranker_cache_for_tests",
]
