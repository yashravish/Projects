"""Model registry — durable storage of *artifacts* indexed by name + version.

This is independent of the SQL `RegisteredModel` table, which is the
*relational* lineage record. The registry lives at the artifact layer:

    Local:       /data/models/{name}/{version}/
                    ├── model.joblib
                    ├── manifest.json
                    └── metrics.json

    SageMaker:   ModelPackageGroup(name) + ModelPackages indexed by version.

The two implementations satisfy the same `ModelRegistry` Protocol so the
service layer treats them identically. The local path is durable across
container restarts because it lives on the `models` named volume mounted
into both `api` and `worker` services in `compose.yaml`.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import shutil
from typing import Any, Iterator, Protocol, cast, runtime_checkable

from app.config import Settings, get_settings
from app.ml.boto_typing import Boto3Module, SageMakerModelPackageGroupClient
from app.logging_config import get_logger
from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
)

log = get_logger("ml.registry")

INDEX_FILENAME = "index.json"


@dataclasses.dataclass(frozen=True)
class ArtifactHandle:
    """Just enough to load a fitted classifier from the registry."""

    name: str
    version: str
    artifact_uri: str
    """Local filesystem path or s3:// URI."""
    local_dir: str
    """Where the model.joblib + manifest.json live on local disk.

    For the local registry this is identical to artifact_uri sans `file://`.
    For the SageMaker registry this is a temp-cached download of the model
    tar.gz contents."""
    framework: str
    framework_version: str
    metrics: dict[str, float | int]
    manifest: dict[str, Any]


@runtime_checkable
class ModelRegistry(Protocol):
    """The contract every registry must satisfy."""

    name: str

    def register(
        self,
        *,
        name: str,
        version: str,
        local_dir: str,
        artifact_uri: str,
        manifest: dict[str, Any],
        metrics: dict[str, float | int],
    ) -> ArtifactHandle:  # pragma: no cover — Protocol declaration
        ...

    def resolve(
        self, *, name: str, version: str
    ) -> ArtifactHandle:  # pragma: no cover
        ...


# ── Local registry ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class LocalModelRegistry:
    """Filesystem-backed registry rooted at `Settings.models_dir`.

    The on-disk layout doubles as the registry index — a `index.json` at
    each model's root tracks the set of known versions and a small
    `last_registered_at` stamp. The SQL `RegisteredModel` table holds
    everything else (stage, lineage to TrainingJob, tenant scoping) so we
    don't duplicate that here; the SQL layer is the source of truth for
    business state, the filesystem for *bytes*.
    """

    base_dir: str
    name: str = "local"

    @property
    def root(self) -> pathlib.Path:
        return pathlib.Path(self.base_dir).expanduser().resolve()

    def _model_dir(self, name: str) -> pathlib.Path:
        return self.root / name

    def _version_dir(self, name: str, version: str) -> pathlib.Path:
        return self._model_dir(name) / version

    def register(
        self,
        *,
        name: str,
        version: str,
        local_dir: str,
        artifact_uri: str,
        manifest: dict[str, Any],
        metrics: dict[str, float | int],
    ) -> ArtifactHandle:
        target = self._version_dir(name, version)

        src = pathlib.Path(local_dir).expanduser().resolve()
        if src.resolve() != target.resolve():
            target.mkdir(parents=True, exist_ok=True)
            for filename in (MODEL_FILENAME, MANIFEST_FILENAME, METRICS_FILENAME):
                src_file = src / filename
                if src_file.is_file():
                    shutil.copy2(src_file, target / filename)
            for sidecar in ("training_data.jsonl", "training.log"):
                src_file = src / sidecar
                if src_file.is_file():
                    shutil.copy2(src_file, target / sidecar)
        else:
            target.mkdir(parents=True, exist_ok=True)

        # Update the per-name index.json.
        index_path = self._model_dir(name) / INDEX_FILENAME
        index = _read_index(index_path)
        index["versions"][version] = {
            "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "artifact_uri": artifact_uri,
            "framework": str(manifest.get("framework") or "unknown"),
            "framework_version": str(manifest.get("framework_version") or "unknown"),
            "metrics": dict(metrics),
        }
        index["last_registered_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_index(index_path, index)

        log.info(
            "ml.registry.registered",
            backend=self.name,
            name=name,
            version=version,
            local_dir=str(target),
        )
        return ArtifactHandle(
            name=name,
            version=version,
            artifact_uri=artifact_uri or f"file://{target}",
            local_dir=str(target),
            framework=str(manifest.get("framework") or "unknown"),
            framework_version=str(manifest.get("framework_version") or "unknown"),
            metrics=dict(metrics),
            manifest=manifest,
        )

    def resolve(self, *, name: str, version: str) -> ArtifactHandle:
        target = self._version_dir(name, version)
        if not (target / MODEL_FILENAME).is_file():
            raise FileNotFoundError(
                f"no registered artifact for {name}@{version} at {target!s}"
            )
        manifest = _read_optional_json(target / MANIFEST_FILENAME) or {}
        metrics = _read_optional_json(target / METRICS_FILENAME) or {}
        return ArtifactHandle(
            name=name,
            version=version,
            artifact_uri=f"file://{target}",
            local_dir=str(target),
            framework=str(manifest.get("framework") or "unknown"),
            framework_version=str(manifest.get("framework_version") or "unknown"),
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            manifest=manifest,
        )

    def iter_versions(self, name: str) -> Iterator[str]:
        d = self._model_dir(name)
        if not d.is_dir():
            return iter(())
        return (
            child.name
            for child in sorted(d.iterdir())
            if child.is_dir() and (child / MODEL_FILENAME).is_file()
        )


# ── SageMaker registry ───────────────────────────────────────────────────────


@dataclasses.dataclass
class SageMakerModelRegistry:
    """SageMaker Model Package Group registry.

    Each logical model name maps to a ModelPackageGroup; each version is a
    ModelPackage with the artifact URI pointing at the trained model in S3.
    For local resolution we keep a *cache* of the downloaded artifact at
    `cache_dir` so inference doesn't have to re-download every cold start.
    """

    settings: Settings
    cache_dir: str
    name: str = "sagemaker"
    boto3_module: Boto3Module | None = None

    def _boto3(self) -> Boto3Module:
        if self.boto3_module is not None:
            return self.boto3_module
        import boto3

        return cast(Boto3Module, boto3)

    def _ensure_group(
        self, sm_client: SageMakerModelPackageGroupClient, name: str
    ) -> None:
        try:
            sm_client.describe_model_package_group(ModelPackageGroupName=name)
            return
        except Exception:  # noqa: BLE001 — paper over ResourceNotFound
            try:
                sm_client.create_model_package_group(
                    ModelPackageGroupName=name,
                    ModelPackageGroupDescription=(
                        "PSDI cross-encoder reranker — auto-created."
                    ),
                )
            except Exception as create_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"could not ensure model package group {name!r}: {create_exc!s}"
                ) from create_exc

    def register(
        self,
        *,
        name: str,
        version: str,
        local_dir: str,
        artifact_uri: str,
        manifest: dict[str, Any],
        metrics: dict[str, float | int],
    ) -> ArtifactHandle:
        boto3 = self._boto3()
        sm_client = cast(
            SageMakerModelPackageGroupClient,
            boto3.client("sagemaker", region_name=self.settings.aws_region),
        )
        self._ensure_group(sm_client, name)

        if not artifact_uri.startswith("s3://"):
            raise RuntimeError(
                "SageMaker registry requires an s3:// artifact URI; got "
                f"{artifact_uri!r}"
            )

        sm_client.create_model_package(
            ModelPackageGroupName=name,
            ModelPackageDescription=(
                f"version {version} — framework={manifest.get('framework', 'unknown')}"
            ),
            ModelApprovalStatus="Approved",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": _DEFAULT_INFERENCE_IMAGE.format(
                            region=self.settings.aws_region or "us-east-1"
                        ),
                        "ModelDataUrl": artifact_uri,
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            CustomerMetadataProperties={
                "version": version,
                "framework": str(manifest.get("framework") or "unknown"),
                "fingerprint": str(
                    manifest.get("training_data", {}).get("fingerprint") or ""
                ),
            },
        )
        # Mirror the artifacts locally for fast inference.
        cache = pathlib.Path(self.cache_dir) / name / version
        cache.mkdir(parents=True, exist_ok=True)
        for filename in (MODEL_FILENAME, MANIFEST_FILENAME, METRICS_FILENAME):
            src = pathlib.Path(local_dir) / filename
            if src.is_file():
                shutil.copy2(src, cache / filename)

        log.info(
            "ml.registry.sagemaker.registered",
            name=name,
            version=version,
            artifact_uri=artifact_uri,
        )
        return ArtifactHandle(
            name=name,
            version=version,
            artifact_uri=artifact_uri,
            local_dir=str(cache),
            framework=str(manifest.get("framework") or "unknown"),
            framework_version=str(manifest.get("framework_version") or "unknown"),
            metrics=dict(metrics),
            manifest=manifest,
        )

    def resolve(self, *, name: str, version: str) -> ArtifactHandle:
        cache = pathlib.Path(self.cache_dir) / name / version
        if (cache / MODEL_FILENAME).is_file():
            manifest = _read_optional_json(cache / MANIFEST_FILENAME) or {}
            metrics = _read_optional_json(cache / METRICS_FILENAME) or {}
            return ArtifactHandle(
                name=name,
                version=version,
                artifact_uri=f"file://{cache}",
                local_dir=str(cache),
                framework=str(manifest.get("framework") or "unknown"),
                framework_version=str(manifest.get("framework_version") or "unknown"),
                metrics={
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
                manifest=manifest,
            )
        raise FileNotFoundError(
            f"no cached artifact for {name}@{version} at {cache!s}"
        )


_DEFAULT_INFERENCE_IMAGE = (
    "246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_index(path: pathlib.Path) -> dict[str, Any]:
    if path.is_file():
        try:
            data: object = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
    return {"versions": {}, "last_registered_at": None}


def _write_index(path: pathlib.Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")


def _read_optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return cast(dict[str, Any], data)


# ── Factory ──────────────────────────────────────────────────────────────────


def build_model_registry(
    *,
    settings: Settings | None = None,
    boto3_module: Boto3Module | None = None,
) -> ModelRegistry:
    s = settings or get_settings()
    if s.reranker_backend == "sagemaker" and s.sagemaker_role_arn:
        return SageMakerModelRegistry(
            settings=s,
            cache_dir=s.models_dir,
            boto3_module=boto3_module,
        )
    return LocalModelRegistry(base_dir=s.models_dir)


__all__ = [
    "ArtifactHandle",
    "INDEX_FILENAME",
    "LocalModelRegistry",
    "ModelRegistry",
    "SageMakerModelRegistry",
    "build_model_registry",
]
