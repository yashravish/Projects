"""Pydantic shapes for the training & model registry API.

The shapes mirror the SQL rows but always include a flattened `metrics`
dict so the frontend does not have to unpack a JSONB blob in two different
places. Datetimes are serialised in ISO 8601 with timezone.
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelStage = Literal["staging", "production", "archived"]
JobStatus = Literal["pending", "running", "success", "failed"]


class TrainingJobRequest(BaseModel):
    """Body for POST /training/jobs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="psdi-cross-encoder-reranker",
        min_length=1,
        max_length=200,
        description="Logical model name. The trained artifact is registered "
        "under this name; promoting a new version replaces the old one.",
    )
    notes: str | None = Field(
        default=None,
        max_length=2_000,
        description="Free-text note attached to the resulting RegisteredModel row.",
    )
    auto_promote: bool = Field(
        default=False,
        description="If True, automatically promote the model to production on "
        "successful training. Existing production model is archived.",
    )


class TrainingJobMetricsOut(BaseModel):
    """A flat view of the model metrics dict."""

    model_config = ConfigDict(extra="allow")

    n_train: int = 0
    n_holdout: int = 0
    train_accuracy: float = 0.0
    holdout_accuracy: float = 0.0
    holdout_precision: float = 0.0
    holdout_recall: float = 0.0
    holdout_f1: float = 0.0
    holdout_roc_auc: float = 0.0
    holdout_avg_precision: float = 0.0
    holdout_log_loss: float = 0.0
    score_separation: float = 0.0


class TrainingJobSummary(BaseModel):
    """Row in the jobs roll."""

    job_id: uuid.UUID
    name: str
    version: str
    backend: str
    framework: str
    status: JobStatus
    duration_s: float
    holdout_f1: float = 0.0
    holdout_roc_auc: float = 0.0
    score_separation: float = 0.0
    n_train: int = 0
    error_message: str | None = None
    mlflow_run_id: str | None = None
    created_at: dt.datetime


class TrainingJobDetail(BaseModel):
    """Full training-job detail."""

    job_id: uuid.UUID
    organization_id: uuid.UUID
    triggered_by: uuid.UUID | None
    name: str
    version: str
    backend: str
    framework: str
    framework_version: str | None
    status: JobStatus
    artifact_uri: str | None
    external_job_id: str | None
    config: dict[str, Any]
    metrics: TrainingJobMetricsOut
    manifest: dict[str, Any]
    log_excerpt: str | None
    duration_s: float
    mlflow_run_id: str | None
    error_message: str | None
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    created_at: dt.datetime
    registered_model_id: uuid.UUID | None = None


class TrainingJobList(BaseModel):
    items: list[TrainingJobSummary]
    total: int
    page: int
    page_size: int


class RegisteredModelSummary(BaseModel):
    """Row in the registry table."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: uuid.UUID
    name: str
    version: str
    framework: str
    backend: str
    stage: ModelStage
    holdout_f1: float = 0.0
    holdout_roc_auc: float = 0.0
    score_separation: float = 0.0
    n_train: int = 0
    artifact_uri: str
    training_job_id: uuid.UUID | None
    created_at: dt.datetime
    promoted_at: dt.datetime | None
    archived_at: dt.datetime | None


class RegisteredModelDetail(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    version: str
    framework: str
    framework_version: str | None
    backend: str
    artifact_uri: str
    local_dir: str | None
    stage: ModelStage
    metrics: TrainingJobMetricsOut
    manifest: dict[str, Any]
    training_job_id: uuid.UUID | None
    promoted_by: uuid.UUID | None
    notes: str | None
    promoted_at: dt.datetime | None
    archived_at: dt.datetime | None
    created_at: dt.datetime


class RegisteredModelList(BaseModel):
    items: list[RegisteredModelSummary]
    total: int
    page: int
    page_size: int


class PromoteModelRequest(BaseModel):
    """Body for POST /models/{id}/promote."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["production", "archived"] = Field(
        description="Promote to `production` (auto-archives the prior "
        "production model under the same name) or `archived` (decommission)."
    )
    notes: str | None = Field(default=None, max_length=2_000)


class RerankerPredictRequest(BaseModel):
    """Body for POST /models/{id}/predict."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    passages: list[str] = Field(min_length=1, max_length=50)


class ScoredPassage(BaseModel):
    index: int
    passage: str
    score: float


class RerankerPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: uuid.UUID
    model_name: str
    model_version: str
    backend: str
    scored: list[ScoredPassage]
    """Re-ordered: highest score first."""


__all__ = [
    "JobStatus",
    "ModelStage",
    "PromoteModelRequest",
    "RegisteredModelDetail",
    "RegisteredModelList",
    "RegisteredModelSummary",
    "RerankerPredictRequest",
    "RerankerPredictResponse",
    "ScoredPassage",
    "TrainingJobDetail",
    "TrainingJobList",
    "TrainingJobMetricsOut",
    "TrainingJobRequest",
    "TrainingJobSummary",
]
