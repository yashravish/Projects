"""Tests for the local training backend (subprocess) and SageMaker stubbing.

The local backend spawns a real Python subprocess running
`app.ml.training_script`. These tests are slower than the rest of the
unit suite (~3 s) but they're the only way to catch packaging/import
regressions in the training script.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from app.ml.backends import (
    LocalTrainingBackend,
    SageMakerTrainingBackend,
    TrainingJobSpec,
    build_training_backend,
)
from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
)


# ── Local backend ────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_local_backend_runs_training_subprocess(tmp_path: pathlib.Path) -> None:
    backend = LocalTrainingBackend()
    spec = TrainingJobSpec(
        name="psdi-cross-encoder-reranker",
        version="vtest",
        output_dir=str(tmp_path / "artifacts"),
    )
    outcome = backend.run_training_job(spec)
    assert outcome.status == "success", outcome.error_message
    assert outcome.backend == "local"
    out = pathlib.Path(outcome.output_dir)
    assert (out / MODEL_FILENAME).is_file()
    assert (out / MANIFEST_FILENAME).is_file()
    assert (out / METRICS_FILENAME).is_file()
    manifest = json.loads((out / MANIFEST_FILENAME).read_text())
    assert manifest["name"] == spec.name
    assert manifest["version"] == spec.version
    assert outcome.metrics, "metrics should be captured into the outcome"
    assert outcome.duration_s >= 0
    assert outcome.error_message is None


def test_local_backend_marks_failed_when_subprocess_explodes(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """If the subprocess returns non-zero, the outcome must be `failed`."""
    import subprocess

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        # Touch nothing; return a synthetic CompletedProcess with code 17.
        return subprocess.CompletedProcess(
            args=cmd, returncode=17, stdout="", stderr="boom"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = LocalTrainingBackend()
    outcome = backend.run_training_job(
        TrainingJobSpec(
            name="x", version="v", output_dir=str(tmp_path / "out")
        )
    )
    assert outcome.status == "failed"
    assert "boom" in (outcome.error_message or "")


def test_local_backend_handles_timeout(tmp_path: pathlib.Path, monkeypatch) -> None:
    import subprocess

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = LocalTrainingBackend()
    outcome = backend.run_training_job(
        TrainingJobSpec(name="x", version="v", output_dir=str(tmp_path))
    )
    assert outcome.status == "failed"
    assert "timeout" in (outcome.error_message or "").lower()


# ── SageMaker backend (stubbed) ──────────────────────────────────────────────


class _FakeSm:
    def __init__(self, *, completed: bool = True) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.describe_calls: list[dict[str, Any]] = []
        self._sequence = (
            ["InProgress", "InProgress", "Completed" if completed else "Failed"]
        )
        self._idx = 0

    def create_training_job(self, **kwargs: Any) -> dict:
        self.create_calls.append(kwargs)
        return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:0:training-job/x"}

    def describe_training_job(self, **kwargs: Any) -> dict:
        self.describe_calls.append(kwargs)
        status = self._sequence[min(self._idx, len(self._sequence) - 1)]
        self._idx += 1
        return {
            "TrainingJobStatus": status,
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://psdi-models/x/v1/model.tar.gz"
            },
            "FailureReason": "" if status != "Failed" else "synthetic failure",
        }


class _FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def download_fileobj(self, bucket: str, key: str, fh) -> None:  # noqa: ANN001
        self.calls.append((bucket, key))
        # Write an empty file — the test patches _extract_tar so contents
        # don't actually matter.
        fh.write(b"")


class _FakeBoto3SmBackend:
    def __init__(self, sm: _FakeSm, s3: _FakeS3) -> None:
        self._sm = sm
        self._s3 = s3

    def client(self, service_name: str, region_name: str | None = None) -> Any:
        if service_name == "sagemaker":
            return self._sm
        if service_name == "s3":
            return self._s3
        raise AssertionError(f"unexpected client {service_name!r}")


def test_sagemaker_backend_create_then_poll_then_download(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    from app.config import get_settings
    from app.ml import backends as backends_mod

    settings = get_settings()
    monkeypatch.setattr(settings, "sagemaker_role_arn", "arn:aws:iam::0:role/x", raising=False)
    monkeypatch.setattr(settings, "s3_bucket", "psdi-models", raising=False)
    monkeypatch.setattr(settings, "aws_region", "us-east-1", raising=False)

    extracted: list[pathlib.Path] = []

    def fake_extract(tar_path: pathlib.Path, into: pathlib.Path) -> None:
        # Simulate the trained artifacts landing in `into`.
        extracted.append(into)
        for filename in (MODEL_FILENAME, MANIFEST_FILENAME, METRICS_FILENAME):
            (into / filename).write_text(
                json.dumps({"name": "x", "version": "v1"})
                if filename == MANIFEST_FILENAME
                else json.dumps({"holdout_accuracy": 0.9})
                if filename == METRICS_FILENAME
                else "fake"
            )

    monkeypatch.setattr(backends_mod, "_extract_tar", fake_extract)

    sm = _FakeSm(completed=True)
    s3 = _FakeS3()
    backend = SageMakerTrainingBackend(
        settings=settings,
        boto3_module=_FakeBoto3SmBackend(sm, s3),
        poll_interval_s=0.0,  # don't actually sleep in tests
        poll_timeout_s=2.0,
    )
    outcome = backend.run_training_job(
        TrainingJobSpec(
            name="x", version="v1", output_dir=str(tmp_path / "out")
        )
    )
    assert outcome.status == "success", outcome.error_message
    assert outcome.backend == "sagemaker"
    assert sm.create_calls, "should have called create_training_job"
    assert sm.describe_calls, "should have polled describe_training_job"
    assert s3.calls == [("psdi-models", "x/v1/model.tar.gz")]
    assert extracted, "extract_tar should have been invoked"
    assert outcome.external_job_id


def test_sagemaker_backend_marks_failed_status(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sagemaker_role_arn", "arn:x", raising=False)
    monkeypatch.setattr(settings, "s3_bucket", "psdi-models", raising=False)
    monkeypatch.setattr(settings, "aws_region", "us-east-1", raising=False)

    backend = SageMakerTrainingBackend(
        settings=settings,
        boto3_module=_FakeBoto3SmBackend(_FakeSm(completed=False), _FakeS3()),
        poll_interval_s=0.0,
        poll_timeout_s=2.0,
    )
    outcome = backend.run_training_job(
        TrainingJobSpec(name="x", version="v1", output_dir=str(tmp_path))
    )
    assert outcome.status == "failed"
    assert "synthetic failure" in (outcome.error_message or "")


def test_sagemaker_backend_check_config_requires_role(monkeypatch) -> None:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sagemaker_role_arn", "", raising=False)
    backend = SageMakerTrainingBackend(settings=settings)
    with pytest.raises(RuntimeError, match="SAGEMAKER_ROLE_ARN"):
        backend._check_config()


# ── Factory ──────────────────────────────────────────────────────────────────


def test_build_training_backend_falls_back_to_local_when_sagemaker_unconfigured(
    monkeypatch,
) -> None:
    from app.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "reranker_backend", "sagemaker", raising=False)
    monkeypatch.setattr(s, "sagemaker_role_arn", "", raising=False)
    backend = build_training_backend(settings=s)
    assert isinstance(backend, LocalTrainingBackend)


def test_build_training_backend_uses_local_by_default() -> None:
    backend = build_training_backend()
    assert isinstance(backend, LocalTrainingBackend)
