"""Training entry point.

This is the script SageMaker invokes inside its training container — and the
exact same script `LocalTrainingBackend` runs as a subprocess on the host.

Two interfaces:

  1. **CLI** (local backend):
        python -m app.ml.training_script --output-dir /data/models/.../v1

  2. **SageMaker** (SM_MODEL_DIR / SM_OUTPUT_DATA_DIR / SM_CHANNEL_TRAINING):
        python -m app.ml.training_script

When `--training-data` is omitted the script *synthesises* the training
triples in-process from the seeded corpus + gold dataset, ensuring
reproducible runs without an external data-channel.

Outputs:
    {output_dir}/model.joblib       — fitted pipeline
    {output_dir}/manifest.json      — lineage + hyperparams + metrics
    {output_dir}/metrics.json       — metrics-only sidecar
    {output_dir}/training_data.jsonl — the exact rows the model saw
    {output_dir}/training.log       — captured INFO logs
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import pathlib
import platform
import sys
import time
from typing import Any

# Use direct imports rather than the `app.ml` package facade so this script
# can be run inside a SageMaker container that may install only the
# `app.ml` subtree.
from app.ml.classifier import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    TRAINING_DATA_FILENAME,
    CrossEncoderClassifier,
    Hyperparameters,
    TrainingMetrics,
)
from app.ml.training_data import TrainingTriples, synthesize_training_triples

# SageMaker conventions — see
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
SM_MODEL_DIR_ENV = "SM_MODEL_DIR"
SM_OUTPUT_DATA_DIR_ENV = "SM_OUTPUT_DATA_DIR"
SM_CHANNEL_TRAINING_ENV = "SM_CHANNEL_TRAINING"


def _build_logger(log_path: pathlib.Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("psdi.ml.training")
    logger.setLevel(logging.INFO)
    # Clear any inherited handlers (matters when run as a python -m).
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _resolve_output_dir(arg_value: str | None) -> pathlib.Path:
    if arg_value:
        return pathlib.Path(arg_value).expanduser().resolve()
    sm = os.environ.get(SM_MODEL_DIR_ENV)
    if sm:
        return pathlib.Path(sm).expanduser().resolve()
    raise SystemExit(
        "training_script: must supply --output-dir or set SM_MODEL_DIR"
    )


def _resolve_training_data_path(arg_value: str | None) -> pathlib.Path | None:
    if arg_value:
        return pathlib.Path(arg_value).expanduser().resolve()
    sm = os.environ.get(SM_CHANNEL_TRAINING_ENV)
    if sm:
        # SM_CHANNEL_TRAINING is a directory; find the first JSONL inside.
        d = pathlib.Path(sm)
        for child in sorted(d.glob("*.jsonl")):
            return child.resolve()
    return None


def _load_or_synth_triples(
    *,
    path: pathlib.Path | None,
    dataset_name: str | None,
    dataset_version: str | None,
    log: logging.Logger,
) -> TrainingTriples:
    if path is None:
        log.info("training_data.synthesizing")
        return synthesize_training_triples()
    log.info("training_data.loading", extra={"path": str(path)})
    text = path.read_text(encoding="utf-8")
    if dataset_name is None or dataset_version is None:
        # Try to read sidecar metadata.json next to it.
        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dataset_name = dataset_name or str(meta.get("dataset_name", "unknown"))
            dataset_version = dataset_version or str(meta.get("dataset_version", "unknown"))
    return TrainingTriples.from_jsonl(
        text=text,
        dataset_name=dataset_name or "unknown",
        dataset_version=dataset_version or "unknown",
    )


def _build_manifest(
    *,
    name: str,
    version: str,
    framework: str,
    framework_version: str,
    triples: TrainingTriples,
    hyperparameters: Hyperparameters,
    metrics: TrainingMetrics,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "framework": framework,
        "framework_version": framework_version,
        "task": "cross-encoder-reranker",
        "training_data": {
            "rows": len(triples),
            "fingerprint": triples.fingerprint,
            "dataset_name": triples.dataset_name,
            "dataset_version": triples.dataset_version,
            "label_counts": triples.label_counts(),
            "kind_counts": triples.kind_counts(),
        },
        "hyperparameters": hyperparameters.as_dict(),
        "metrics": metrics.as_dict(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round((finished_at - started_at).total_seconds(), 3),
        },
    }
    if extra:
        payload["extra"] = extra
    return payload


def _make_version() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d-%H%M%S")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.ml.training_script",
        description="Train the cross-encoder reranker.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the model artifacts to. Falls back to SM_MODEL_DIR.",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Optional path to a training-data JSONL. If absent, synthesises in-process.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="psdi-cross-encoder-reranker",
        help="Logical model name to record in the manifest.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version string to record. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override dataset_name when loading from --training-data.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Override dataset_version when loading from --training-data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _build_logger(output_dir / "training.log")

    started_at = dt.datetime.now(dt.timezone.utc)
    log.info(f"training.start name={args.name} output_dir={output_dir}")

    try:
        triples = _load_or_synth_triples(
            path=_resolve_training_data_path(args.training_data),
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            log=log,
        )
        log.info(
            f"training_data n_rows={len(triples)} "
            f"label_counts={triples.label_counts()} "
            f"fingerprint={triples.fingerprint}"
        )

        # Persist a copy of the exact training data alongside the model.
        (output_dir / TRAINING_DATA_FILENAME).write_text(
            triples.to_jsonl(), encoding="utf-8"
        )

        hp = Hyperparameters()
        classifier = CrossEncoderClassifier(hyperparameters=hp)
        t0 = time.monotonic()
        metrics = classifier.fit(list(triples.rows))
        elapsed = time.monotonic() - t0
        log.info(
            f"training.fit_complete elapsed_s={elapsed:.3f} "
            f"holdout_f1={metrics.holdout_f1:.3f} "
            f"holdout_roc_auc={metrics.holdout_roc_auc:.3f} "
            f"score_separation={metrics.score_separation:.3f}"
        )

        version = args.version or _make_version()
        finished_at = dt.datetime.now(dt.timezone.utc)
        manifest = _build_manifest(
            name=args.name,
            version=version,
            framework="sklearn-tfidf-logreg",
            framework_version=_sklearn_version(),
            triples=triples,
            hyperparameters=hp,
            metrics=metrics,
            started_at=started_at,
            finished_at=finished_at,
        )

        artifacts = classifier.save(
            output_dir, manifest=manifest, metrics=metrics
        )
        log.info(f"training.save_complete artifacts={artifacts}")
        # Print the manifest path on the last line so subprocess callers can
        # parse it cheaply if they want to. Pure convenience.
        print(str(output_dir / MANIFEST_FILENAME))
        return 0
    except Exception:  # noqa: BLE001 — broad on entrypoint to log everything
        log.exception("training.failed")
        # Write a partial manifest so the subprocess caller can find a record.
        (output_dir / "FAILED").write_text(
            json.dumps(
                {"failed": True, "at": dt.datetime.now(dt.timezone.utc).isoformat()},
                indent=2,
            ),
            encoding="utf-8",
        )
        # NB: do not raise — SageMaker treats nonzero exit as job failure,
        # and the local backend mirrors that.
        return 1


def _sklearn_version() -> str:
    try:
        import sklearn  # local import keeps the module import cheap

        return str(getattr(sklearn, "__version__", "unknown"))
    except (ImportError, AttributeError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]


# Re-export so service code can persist a metrics-key set even on failure.
_ = METRICS_FILENAME, MANIFEST_FILENAME, dataclasses
