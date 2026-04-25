"""MLflow tracking integration.

A thin async-friendly recorder that:
- lazily configures the tracking URI + experiment from `Settings`,
- exposes `record_inquiry()` / `record_evaluation()` context managers,
- never raises into the caller — a tracking outage must not break inquiries.

The MLflow Python SDK is sync. We isolate every blocking call inside the
recorder so the rest of the codebase stays async-clean. Latency-critical
callers can run recordings in a background task.
"""
from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import mlflow

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger("observability.mlflow")


_singleton: MLflowRunRecorder | None = None


@dataclasses.dataclass
class _ActiveRun:
    run_id: str | None
    enabled: bool


class MLflowRunRecorder:
    """Thin wrapper around the MLflow SDK.

    Construction is cheap and side-effect-free; the first call that needs an
    experiment creates it lazily. Failures (network, server down) flip the
    recorder into a NO-OP mode for the remainder of its life so downstream
    callers don't pay repeated retry costs in a hot path.
    """

    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._configured = False
        self._disabled = False

    def _configure(self) -> bool:
        if self._configured:
            return True
        if self._disabled:
            return False
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._configured = True
            log.info(
                "mlflow.configured",
                tracking_uri=self._tracking_uri,
                experiment=self._experiment_name,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            log.warning(
                "mlflow.configure_failed",
                tracking_uri=self._tracking_uri,
                error=str(exc),
            )
            self._disabled = True
            return False

    @contextlib.contextmanager
    def record_inquiry(
        self,
        *,
        organization_id: str,
        run_name: str,
        tags: dict[str, str],
    ) -> Iterator[_ActiveRun]:
        """Open an MLflow run for one inquiry. Yields an `_ActiveRun` whose
        `run_id` is None if MLflow is unavailable.
        """
        if not self._configure():
            yield _ActiveRun(run_id=None, enabled=False)
            return

        merged_tags = {"organization_id": organization_id, **tags}
        active = _ActiveRun(run_id=None, enabled=True)
        try:
            run = mlflow.start_run(run_name=run_name, tags=merged_tags)
            active.run_id = run.info.run_id
            yield active
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow.start_run_failed", error=str(exc))
            yield _ActiveRun(run_id=None, enabled=False)
            return
        finally:
            try:
                if active.run_id is not None:
                    mlflow.end_run()
            except Exception as exc:  # noqa: BLE001
                log.warning("mlflow.end_run_failed", error=str(exc))

    @contextlib.contextmanager
    def record_evaluation(
        self,
        *,
        organization_id: str,
        run_name: str,
        tags: dict[str, str],
    ) -> Iterator[_ActiveRun]:
        """Open an MLflow run for one evaluation harness execution.

        Functionally identical to `record_inquiry` but distinguishes the
        intent in MLflow tags so the UI can filter inquiry vs eval runs.
        """
        merged = {"kind": "evaluation", **tags}
        with self.record_inquiry(
            organization_id=organization_id,
            run_name=run_name,
            tags=merged,
        ) as active:
            yield active

    @contextlib.contextmanager
    def record_training(
        self,
        *,
        organization_id: str,
        run_name: str,
        tags: dict[str, str],
    ) -> Iterator[_ActiveRun]:
        """Open an MLflow run for one training-job execution.

        Tagged `kind=training` so the UI can filter inquiry / eval / training.
        """
        merged = {"kind": "training", **tags}
        with self.record_inquiry(
            organization_id=organization_id,
            run_name=run_name,
            tags=merged,
        ) as active:
            yield active

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        if not self._configured or self._disabled:
            return
        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow.log_artifact_failed", error=str(exc))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self._configured or self._disabled:
            return
        try:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow.log_metrics_failed", error=str(exc))

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._configured or self._disabled:
            return
        try:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow.log_params_failed", error=str(exc))

    def log_dict(self, payload: dict[str, Any], *, artifact_file: str) -> None:
        if not self._configured or self._disabled:
            return
        try:
            mlflow.log_dict(payload, artifact_file=artifact_file)
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow.log_dict_failed", error=str(exc))


def get_mlflow_recorder() -> MLflowRunRecorder:
    global _singleton
    if _singleton is None:
        s = get_settings()
        _singleton = MLflowRunRecorder(
            tracking_uri=s.mlflow_tracking_uri,
            experiment_name=s.mlflow_experiment_name,
        )
    return _singleton


def reset_recorder_for_tests() -> None:
    """Drop the recorder singleton — tests use this to avoid hitting the
    real MLflow server."""
    global _singleton
    _singleton = None


__all__ = [
    "MLflowRunRecorder",
    "get_mlflow_recorder",
    "reset_recorder_for_tests",
]
