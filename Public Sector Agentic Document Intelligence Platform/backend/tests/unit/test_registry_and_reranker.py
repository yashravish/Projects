"""Tests for the local model registry + reranker — pure-Python, no DB."""
from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    CrossEncoderClassifier,
)
from app.ml.registry import (
    INDEX_FILENAME,
    ArtifactHandle,
    LocalModelRegistry,
    SageMakerModelRegistry,
)
from app.ml.reranker import (
    LocalReranker,
    NullReranker,
    RerankRequest,
    SageMakerReranker,
    build_reranker_from_handle,
    reset_reranker_cache_for_tests,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _train_into(target_dir: pathlib.Path, *, name: str = "x", version: str = "v1") -> dict:
    from tests.unit.test_classifier import _toy_examples

    target_dir.mkdir(parents=True, exist_ok=True)
    clf = CrossEncoderClassifier()
    metrics = clf.fit(_toy_examples())
    clf.save(
        target_dir,
        manifest={
            "name": name,
            "version": version,
            "framework": "sklearn-tfidf-logreg",
            "framework_version": "test",
        },
        metrics=metrics,
    )
    return {"framework": "sklearn-tfidf-logreg", "framework_version": "test"}


# ── LocalModelRegistry ──────────────────────────────────────────────────────


def test_local_registry_register_and_resolve(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    _train_into(src, name="psdi-cross-encoder-reranker", version="v1")
    registry = LocalModelRegistry(base_dir=str(tmp_path / "models"))
    handle = registry.register(
        name="psdi-cross-encoder-reranker",
        version="v1",
        local_dir=str(src),
        artifact_uri=f"file://{src}",
        manifest={"framework": "sklearn-tfidf-logreg", "framework_version": "test"},
        metrics={"holdout_accuracy": 0.9},
    )
    assert isinstance(handle, ArtifactHandle)
    target = pathlib.Path(handle.local_dir)
    assert (target / MODEL_FILENAME).is_file()
    assert (target / MANIFEST_FILENAME).is_file()
    assert (target / METRICS_FILENAME).is_file()
    # Index is durable on disk.
    index = json.loads((target.parent / INDEX_FILENAME).read_text())
    assert "v1" in index["versions"]
    # Resolve returns the same artifact.
    resolved = registry.resolve(name="psdi-cross-encoder-reranker", version="v1")
    assert resolved.local_dir == handle.local_dir


def test_local_registry_resolve_missing_raises(tmp_path: pathlib.Path) -> None:
    registry = LocalModelRegistry(base_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        registry.resolve(name="nope", version="v1")


def test_local_registry_register_is_idempotent_for_same_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Calling register twice with the same source must not raise / corrupt."""
    src = tmp_path / "src"
    _train_into(src)
    registry = LocalModelRegistry(base_dir=str(tmp_path / "models"))
    h1 = registry.register(
        name="x",
        version="v1",
        local_dir=str(src),
        artifact_uri=f"file://{src}",
        manifest={"framework": "f"},
        metrics={"a": 0.1},
    )
    h2 = registry.register(
        name="x",
        version="v1",
        local_dir=str(src),
        artifact_uri=f"file://{src}",
        manifest={"framework": "f"},
        metrics={"a": 0.2},
    )
    assert h1.local_dir == h2.local_dir


def test_local_registry_iter_versions(tmp_path: pathlib.Path) -> None:
    registry = LocalModelRegistry(base_dir=str(tmp_path))
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    _train_into(src1)
    _train_into(src2)
    registry.register(
        name="m",
        version="v1",
        local_dir=str(src1),
        artifact_uri=f"file://{src1}",
        manifest={},
        metrics={},
    )
    registry.register(
        name="m",
        version="v2",
        local_dir=str(src2),
        artifact_uri=f"file://{src2}",
        manifest={},
        metrics={},
    )
    assert list(registry.iter_versions("m")) == ["v1", "v2"]


# ── Reranker ────────────────────────────────────────────────────────────────


def test_local_reranker_scores_in_request_order(tmp_path: pathlib.Path) -> None:
    _train_into(tmp_path)
    reset_reranker_cache_for_tests()
    rr = LocalReranker(
        local_dir=str(tmp_path),
        model_name="x",
        model_version="v1",
    )
    out = rr.score(
        RerankRequest(
            query="When is the grant deadline?",
            passages=(
                "Records officers should expect a 12-18% increase.",
                "Applications must be submitted by February 28, 2026.",
                "The procurement vendor portal is available at portal.gov.",
            ),
        )
    )
    assert len(out.scores) == 3
    assert out.backend == "local"
    # The grant-deadline passage (index 1) should out-score the others.
    assert out.scores[1] > out.scores[0]
    assert out.scores[1] > out.scores[2]


def test_local_reranker_handles_empty_passages(tmp_path: pathlib.Path) -> None:
    _train_into(tmp_path)
    reset_reranker_cache_for_tests()
    rr = LocalReranker(
        local_dir=str(tmp_path),
        model_name="x",
        model_version="v1",
    )
    out = rr.score(RerankRequest(query="anything", passages=()))
    assert out.scores == ()


def test_null_reranker_preserves_size_and_uniform() -> None:
    nr = NullReranker()
    out = nr.score(RerankRequest(query="q", passages=("a", "b", "c")))
    assert len(out.scores) == 3
    assert all(s == 0.5 for s in out.scores)
    assert out.backend == "null"


def test_build_reranker_local_by_default(tmp_path: pathlib.Path) -> None:
    rr = build_reranker_from_handle(
        name="x",
        version="v1",
        local_dir=str(tmp_path),
    )
    assert isinstance(rr, LocalReranker)


def test_build_reranker_sagemaker_when_endpoint_configured() -> None:
    from app.config import get_settings

    base = get_settings()
    s = base.model_copy(
        update={
            "reranker_backend": "sagemaker",
            "sagemaker_reranker_endpoint": "psdi-rerank-prod",
            "aws_region": "us-east-1",
        }
    )
    rr = build_reranker_from_handle(
        name="x", version="v1", local_dir="/tmp/x", settings=s
    )
    assert isinstance(rr, SageMakerReranker)
    assert rr.endpoint_name == "psdi-rerank-prod"


# ── SageMaker reranker ──────────────────────────────────────────────────────


class _FakeSagemakerRuntime:
    def __init__(self, scores: list[float]):
        self._scores = scores
        self.last_call: dict[str, Any] | None = None

    def invoke_endpoint(self, **kwargs: Any) -> dict:
        self.last_call = kwargs
        body = json.dumps({"scores": self._scores}).encode("utf-8")

        class _BodyReader:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _BodyReader(body)}


class _FakeBoto3:
    def __init__(self, runtime: _FakeSagemakerRuntime):
        self._runtime = runtime

    def client(self, service_name: str, region_name: str | None = None) -> Any:
        assert service_name == "sagemaker-runtime"
        return self._runtime


def test_sagemaker_reranker_invokes_endpoint_with_correct_payload() -> None:
    from app.config import get_settings

    runtime = _FakeSagemakerRuntime([0.1, 0.9])
    rr = SageMakerReranker(
        settings=get_settings(),
        endpoint_name="psdi-rerank",
        model_name="x",
        model_version="v1",
        boto3_module=_FakeBoto3(runtime),
    )
    out = rr.score(RerankRequest(query="q", passages=("a", "b")))
    assert out.scores == (0.1, 0.9)
    assert out.backend == "sagemaker"
    assert runtime.last_call is not None
    assert runtime.last_call["EndpointName"] == "psdi-rerank"
    body = json.loads(runtime.last_call["Body"].decode("utf-8"))
    assert body == {"query": "q", "passages": ["a", "b"]}


def test_sagemaker_reranker_rejects_malformed_response() -> None:
    from app.config import get_settings

    runtime = _FakeSagemakerRuntime([0.5])  # length mismatch
    rr = SageMakerReranker(
        settings=get_settings(),
        endpoint_name="psdi",
        model_name="x",
        model_version="v1",
        boto3_module=_FakeBoto3(runtime),
    )
    with pytest.raises(RuntimeError, match="malformed"):
        rr.score(RerankRequest(query="q", passages=("a", "b")))


# ── SageMaker registry (stubbed) ─────────────────────────────────────────────


class _FakeSagemakerControl:
    """Minimal stub of the sagemaker control-plane API."""

    def __init__(self) -> None:
        self.created_groups: list[dict] = []
        self.created_packages: list[dict] = []
        self._groups: set[str] = set()

    def describe_model_package_group(self, **kwargs: Any) -> dict:
        if kwargs["ModelPackageGroupName"] not in self._groups:
            raise RuntimeError("ResourceNotFound")
        return {"ModelPackageGroupName": kwargs["ModelPackageGroupName"]}

    def create_model_package_group(self, **kwargs: Any) -> dict:
        self._groups.add(kwargs["ModelPackageGroupName"])
        self.created_groups.append(kwargs)
        return kwargs

    def create_model_package(self, **kwargs: Any) -> dict:
        self.created_packages.append(kwargs)
        return kwargs


class _FakeBoto3SageMaker:
    def __init__(self, control: _FakeSagemakerControl) -> None:
        self.control = control

    def client(self, service_name: str, region_name: str | None = None) -> Any:
        assert service_name == "sagemaker"
        return self.control


def test_sagemaker_registry_creates_group_and_package(tmp_path: pathlib.Path) -> None:
    from app.config import get_settings

    src = tmp_path / "src"
    _train_into(src, name="rr", version="v42")

    control = _FakeSagemakerControl()
    reg = SageMakerModelRegistry(
        settings=get_settings(),
        cache_dir=str(tmp_path / "cache"),
        boto3_module=_FakeBoto3SageMaker(control),
    )
    handle = reg.register(
        name="rr",
        version="v42",
        local_dir=str(src),
        artifact_uri="s3://psdi-models/rr/v42/model.tar.gz",
        manifest={"framework": "sklearn-tfidf-logreg", "framework_version": "test"},
        metrics={"holdout_accuracy": 0.9},
    )
    assert handle.artifact_uri == "s3://psdi-models/rr/v42/model.tar.gz"
    assert handle.local_dir.startswith(str(tmp_path / "cache"))
    # Group was auto-created and a package registered.
    assert control.created_groups
    assert control.created_packages
    pkg = control.created_packages[0]
    assert pkg["ModelPackageGroupName"] == "rr"
    assert pkg["CustomerMetadataProperties"]["version"] == "v42"
    # Local cache file is mirrored.
    assert (
        pathlib.Path(handle.local_dir) / MODEL_FILENAME
    ).is_file(), "artifact should be cached locally for inference"


def test_sagemaker_registry_rejects_non_s3_artifact(tmp_path: pathlib.Path) -> None:
    from app.config import get_settings

    src = tmp_path / "src"
    _train_into(src)
    reg = SageMakerModelRegistry(
        settings=get_settings(),
        cache_dir=str(tmp_path / "cache"),
        boto3_module=_FakeBoto3SageMaker(_FakeSagemakerControl()),
    )
    with pytest.raises(RuntimeError, match="s3://"):
        reg.register(
            name="rr",
            version="v1",
            local_dir=str(src),
            artifact_uri=f"file://{src}",
            manifest={},
            metrics={},
        )
