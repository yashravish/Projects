"""Pluggable training backends.

Two implementations behind a single Protocol:

    * `LocalTrainingBackend`     — subprocess-based, runs entirely on the
                                    host. Always available, used in tests
                                    and as the default in development.

    * `SageMakerTrainingBackend` — submits a SageMaker training job via
                                    boto3. Activated when
                                    `Settings.reranker_backend == "sagemaker"`
                                    and AWS credentials are present.

Both produce a `TrainingJobOutcome` whose shape is identical regardless of
where the job actually ran. The service layer (`training_service.py`)
operates only against this protocol — it never imports either implementation
directly — so promoting a model trained on SageMaker is identical to
promoting one trained locally.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.ml.boto_typing import Boto3Module, S3GetClient, SageMakerTrainingClient
from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    TRAINING_DATA_FILENAME,
)

log = get_logger("ml.backends")


# ── Shared types ─────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TrainingJobSpec:
    """A request to train one model. Backend-agnostic."""

    name: str
    """Logical model name to record in the manifest (e.g. `psdi-cross-encoder-reranker`)."""
    version: str
    """Caller-chosen version string (e.g. `v20260425-153012-abc123`)."""
    output_dir: str
    """Local directory the artifacts must end up in.

    For SageMaker this is the directory we'll *download to* once the job
    finishes; for the local backend it's where the subprocess writes
    directly. Either way the manifest sees this path, so the registry can
    index from a single source of truth."""
    extra_env: dict[str, str] = dataclasses.field(default_factory=dict)
    """Extra environment variables to pass into the training process."""


@dataclasses.dataclass(frozen=True)
class TrainingJobOutcome:
    """The result of one training job. Persisted on `TrainingJob` rows."""

    backend: str
    """`"local"` | `"sagemaker"`."""
    status: str
    """`"success"` | `"failed"`."""
    output_dir: str
    """Where the artifacts ended up on the local filesystem.

    For SageMaker this is *after* artifact retrieval — the registry only
    needs to know the local path."""
    artifact_uri: str
    """Canonical URI: a `file://` path for local; an `s3://...` URI when the
    backend uploaded the artifact to S3 (SageMaker's Model Package Group
    requires this)."""
    manifest: dict[str, Any]
    """The full contents of `manifest.json`."""
    metrics: dict[str, float | int]
    """The contents of `metrics.json` parsed into a flat dict."""
    framework: str
    framework_version: str
    started_at: dt.datetime
    finished_at: dt.datetime
    duration_s: float
    log_excerpt: str
    """Last ~4 KB of training log; useful for surfacing in the API."""
    external_job_id: str | None = None
    """SageMaker training job ARN (None for local)."""
    error_message: str | None = None


@runtime_checkable
class TrainingBackend(Protocol):
    """The contract every training backend must satisfy."""

    name: str

    def run_training_job(
        self, spec: TrainingJobSpec
    ) -> TrainingJobOutcome:  # pragma: no cover — Protocol declaration
        ...


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return cast(dict[str, Any], data)


def _timeout_log_excerpt(
    log_path: pathlib.Path, stdout: str | bytes | None
) -> str:
    head = _read_log_excerpt(log_path)
    if head:
        return head
    if stdout is None:
        return ""
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")[-4096:]
    return stdout[-4096:]


def _read_log_excerpt(path: pathlib.Path, *, max_bytes: int = 4096) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - max_bytes)
            fh.seek(start)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _flat_numeric(d: dict[str, Any]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = v
    return out


# ── Local backend ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class LocalTrainingBackend:
    """Run the training script as an in-process subprocess.

    The subprocess inherits PYTHONPATH so it can resolve `app.ml...`. The
    output directory is created if missing. Stdout/stderr are streamed into
    `training.log` inside the output dir (the script does this itself) and
    the tail is also embedded in the outcome.
    """

    settings: Settings | None = None
    name: str = "local"

    def _python(self) -> str:
        # `sys.executable` is the running interpreter; identical to what
        # the SageMaker container would invoke.
        return sys.executable

    def run_training_job(self, spec: TrainingJobSpec) -> TrainingJobOutcome:
        output_dir = pathlib.Path(spec.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            **spec.extra_env,
            # Don't let this subprocess re-trigger seed/migrations.
            "SEED_ON_BOOT": "false",
            "RUN_MIGRATIONS": "false",
            # Buffering off so training.log is flushed promptly.
            "PYTHONUNBUFFERED": "1",
        }
        cmd = [
            self._python(),
            "-m",
            "app.ml.training_script",
            "--output-dir",
            str(output_dir),
            "--name",
            spec.name,
            "--version",
            spec.version,
        ]
        log.info(
            "ml.local_backend.start",
            name=spec.name,
            version=spec.version,
            output_dir=str(output_dir),
        )
        started_at = dt.datetime.now(dt.timezone.utc)
        t0 = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 — args are not user-supplied
                cmd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute hard ceiling for the toy classifier
            )
            duration = time.monotonic() - t0
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            finished_at = dt.datetime.now(dt.timezone.utc)
            log.error(
                "ml.local_backend.timeout",
                name=spec.name,
                version=spec.version,
                output_dir=str(output_dir),
            )
            return TrainingJobOutcome(
                backend=self.name,
                status="failed",
                output_dir=str(output_dir),
                artifact_uri=f"file://{output_dir}",
                manifest={},
                metrics={},
                framework="sklearn-tfidf-logreg",
                framework_version="unknown",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration,
                log_excerpt=_timeout_log_excerpt(
                    output_dir / "training.log", exc.stdout
                ),
                error_message=f"timeout after {exc.timeout}s",
            )

        finished_at = dt.datetime.now(dt.timezone.utc)
        manifest = _read_optional_json(output_dir / MANIFEST_FILENAME) or {}
        metrics_raw = _read_optional_json(output_dir / METRICS_FILENAME) or {}
        metrics = _flat_numeric(metrics_raw)
        log_excerpt = _read_log_excerpt(output_dir / "training.log")
        if not log_excerpt:
            log_excerpt = (completed.stdout or "")[-4096:] + "\n" + (completed.stderr or "")[-4096:]

        success = (
            completed.returncode == 0
            and (output_dir / MODEL_FILENAME).is_file()
            and bool(manifest)
            and not (output_dir / "FAILED").exists()
        )
        status = "success" if success else "failed"
        error_message = (
            None
            if success
            else (completed.stderr or "training script exited with non-zero status").strip()[:600]
        )
        framework = str(manifest.get("framework") or "sklearn-tfidf-logreg")
        framework_version = str(manifest.get("framework_version") or "unknown")

        return TrainingJobOutcome(
            backend=self.name,
            status=status,
            output_dir=str(output_dir),
            artifact_uri=f"file://{output_dir}",
            manifest=manifest,
            metrics=metrics,
            framework=framework,
            framework_version=framework_version,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration,
            log_excerpt=log_excerpt,
            external_job_id=None,
            error_message=error_message,
        )


# ── SageMaker backend ────────────────────────────────────────────────────────


_DEFAULT_SAGEMAKER_INSTANCE = "ml.m5.large"
_DEFAULT_SAGEMAKER_VOLUME_GB = 10
_DEFAULT_SAGEMAKER_MAX_RUNTIME_S = 1800
_DEFAULT_SAGEMAKER_FRAMEWORK_IMAGE = (
    # Public AWS-managed sklearn 1.5 CPU image. Region-substitutable.
    "246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
)


@dataclasses.dataclass
class SageMakerTrainingBackend:
    """Submit a real SageMaker training job, then download the artifacts.

    The contract: this backend NEVER fakes a successful job. If credentials,
    role, or image are missing it raises early so the service can fail the
    job cleanly and the operator gets a useful error in the UI. Tests use
    the boto3 stubber to simulate cloud responses.

    We deliberately keep the implementation minimal — the platform's primary
    runtime is local — but the loop is real: `create_training_job` →
    `describe_training_job` poll → S3 model download → manifest read →
    register Model Package.
    """

    settings: Settings
    name: str = "sagemaker"
    poll_interval_s: float = 10.0
    poll_timeout_s: float = float(_DEFAULT_SAGEMAKER_MAX_RUNTIME_S + 600)
    boto3_module: Boto3Module | None = None  # injectable for testing
    s3_bucket: str | None = None
    framework_image: str | None = None
    instance_type: str = _DEFAULT_SAGEMAKER_INSTANCE
    volume_size_gb: int = _DEFAULT_SAGEMAKER_VOLUME_GB
    max_runtime_s: int = _DEFAULT_SAGEMAKER_MAX_RUNTIME_S

    def _check_config(self) -> None:
        if not self.settings.sagemaker_role_arn:
            raise RuntimeError(
                "SageMaker training backend requires SAGEMAKER_ROLE_ARN"
            )
        bucket = self.s3_bucket or self.settings.s3_bucket
        if not bucket:
            raise RuntimeError(
                "SageMaker training backend requires an S3 bucket "
                "(set S3_BUCKET in the environment)"
            )

    def _boto3(self) -> Boto3Module:
        if self.boto3_module is not None:
            return self.boto3_module
        import boto3  # lazy: keeps the import out of the cold start path

        return cast(Boto3Module, boto3)

    def _resolved_image(self) -> str:
        if self.framework_image:
            return self.framework_image
        return _DEFAULT_SAGEMAKER_FRAMEWORK_IMAGE.format(
            region=self.settings.aws_region or "us-east-1"
        )

    def run_training_job(self, spec: TrainingJobSpec) -> TrainingJobOutcome:
        self._check_config()

        boto3 = self._boto3()
        sm_client = cast(
            SageMakerTrainingClient,
            boto3.client("sagemaker", region_name=self.settings.aws_region),
        )
        s3_client = cast(
            S3GetClient,
            boto3.client("s3", region_name=self.settings.aws_region),
        )
        bucket = self.s3_bucket or self.settings.s3_bucket
        prefix = f"models/{spec.name}/{spec.version}"
        s3_output_uri = f"s3://{bucket}/{prefix}/output/"
        sagemaker_job_name = (
            f"{spec.name}-{spec.version}".replace("_", "-").lower()
        )[:63]

        started_at = dt.datetime.now(dt.timezone.utc)
        t0 = time.monotonic()

        try:
            sm_client.create_training_job(
                TrainingJobName=sagemaker_job_name,
                RoleArn=self.settings.sagemaker_role_arn,
                AlgorithmSpecification={
                    "TrainingImage": self._resolved_image(),
                    "TrainingInputMode": "File",
                },
                ResourceConfig={
                    "InstanceType": self.instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": self.volume_size_gb,
                },
                StoppingCondition={
                    "MaxRuntimeInSeconds": self.max_runtime_s,
                },
                OutputDataConfig={"S3OutputPath": s3_output_uri},
                HyperParameters={
                    "name": spec.name,
                    "version": spec.version,
                },
                Environment=spec.extra_env,
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - t0
            finished_at = dt.datetime.now(dt.timezone.utc)
            log.exception(
                "ml.sagemaker_backend.create_failed",
                name=spec.name,
                version=spec.version,
            )
            return TrainingJobOutcome(
                backend=self.name,
                status="failed",
                output_dir=spec.output_dir,
                artifact_uri=s3_output_uri,
                manifest={},
                metrics={},
                framework="sklearn-tfidf-logreg",
                framework_version="unknown",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration,
                log_excerpt="",
                external_job_id=sagemaker_job_name,
                error_message=f"create_training_job failed: {exc!s}"[:600],
            )

        # Poll until terminal status.
        deadline = time.monotonic() + self.poll_timeout_s
        terminal_status: str = "InProgress"
        last_description: dict[str, Any] = {}
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_s)
            desc: Mapping[str, Any] = sm_client.describe_training_job(
                TrainingJobName=sagemaker_job_name
            )
            last_description = cast(dict[str, Any], desc)
            terminal_status = str(last_description.get("TrainingJobStatus", "InProgress"))
            if terminal_status in ("Completed", "Failed", "Stopped"):
                break

        finished_at = dt.datetime.now(dt.timezone.utc)
        duration = time.monotonic() - t0

        if terminal_status != "Completed":
            failure_reason = str(last_description.get("FailureReason", terminal_status))
            return TrainingJobOutcome(
                backend=self.name,
                status="failed",
                output_dir=spec.output_dir,
                artifact_uri=s3_output_uri,
                manifest={},
                metrics={},
                framework="sklearn-tfidf-logreg",
                framework_version="unknown",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration,
                log_excerpt="",
                external_job_id=sagemaker_job_name,
                error_message=f"sagemaker job ended {terminal_status}: {failure_reason}"[:600],
            )

        # Download the model.tar.gz from the configured output path.
        # SageMaker actually places it at s3://{bucket}/{prefix}/output/{job}/output/model.tar.gz
        model_artifact = (
            last_description.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        )
        local_dir = pathlib.Path(spec.output_dir).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            if model_artifact and model_artifact.startswith("s3://"):
                _, _, rest = model_artifact.partition("s3://")
                bkt, _, key = rest.partition("/")
                tar_path = local_dir / "model.tar.gz"
                with tar_path.open("wb") as fh:
                    s3_client.download_fileobj(bkt, key, fh)
                _extract_tar(tar_path, local_dir)
        except Exception as exc:  # noqa: BLE001
            return TrainingJobOutcome(
                backend=self.name,
                status="failed",
                output_dir=str(local_dir),
                artifact_uri=model_artifact or s3_output_uri,
                manifest={},
                metrics={},
                framework="sklearn-tfidf-logreg",
                framework_version="unknown",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration,
                log_excerpt="",
                external_job_id=sagemaker_job_name,
                error_message=f"artifact download failed: {exc!s}"[:600],
            )

        manifest = _read_optional_json(local_dir / MANIFEST_FILENAME) or {}
        metrics_raw = _read_optional_json(local_dir / METRICS_FILENAME) or {}
        metrics = _flat_numeric(metrics_raw)
        log_excerpt = _read_log_excerpt(local_dir / "training.log")
        framework = str(manifest.get("framework") or "sklearn-tfidf-logreg")
        framework_version = str(manifest.get("framework_version") or "unknown")

        return TrainingJobOutcome(
            backend=self.name,
            status="success",
            output_dir=str(local_dir),
            artifact_uri=model_artifact or s3_output_uri,
            manifest=manifest,
            metrics=metrics,
            framework=framework,
            framework_version=framework_version,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration,
            log_excerpt=log_excerpt,
            external_job_id=sagemaker_job_name,
        )


def _extract_tar(tar_path: pathlib.Path, into: pathlib.Path) -> None:
    """Defensive tar extraction.

    Refuses any tar member whose resolved path escapes `into`; this guards
    against a malicious training image producing a tarball with `..` paths.
    """
    import tarfile

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (into / member.name).resolve()
            try:
                target.relative_to(into.resolve())
            except ValueError:
                raise RuntimeError(
                    f"refusing to extract suspicious tar member: {member.name}"
                ) from None
        tar.extractall(into)
    # Best effort cleanup of the now-redundant tarball.
    try:
        tar_path.unlink(missing_ok=True)
    except OSError:
        pass


# ── Factory ──────────────────────────────────────────────────────────────────


def build_training_backend(
    *,
    settings: Settings | None = None,
    boto3_module: Boto3Module | None = None,
) -> TrainingBackend:
    """Pick a backend per `Settings.reranker_backend`.

    Falls back to local when the user asks for sagemaker but the AWS
    configuration is incomplete; we log a warning so the operator sees it.
    """
    s = settings or get_settings()
    if s.reranker_backend == "sagemaker":
        try:
            be = SageMakerTrainingBackend(settings=s, boto3_module=boto3_module)
            be._check_config()
            return be
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "ml.training_backend.sagemaker_unavailable",
                error=str(exc),
            )
    return LocalTrainingBackend(settings=s)


__all__ = [
    "LocalTrainingBackend",
    "SageMakerTrainingBackend",
    "TRAINING_DATA_FILENAME",
    "TrainingBackend",
    "TrainingJobOutcome",
    "TrainingJobSpec",
    "build_training_backend",
]
