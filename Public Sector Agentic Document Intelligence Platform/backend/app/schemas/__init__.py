"""Pydantic v2 DTOs for the public API.

Domain code lives in submodules (`app.schemas.auth`, `app.schemas.documents`, …).

This package re-exports modules that are cheap to import at load time. The
`query` and `evaluation` submodules pull in agent / eval graphs and optional
heavy dependencies; import them explicitly when needed, e.g.
`from app.schemas import query` or `from app.schemas.evaluation import …`.
"""
from __future__ import annotations

from app.schemas.audit import (
    AuditEventFilters,
    AuditEventList,
    AuditEventOut,
    AuditOutcome,
    IntegrityBreak,
    IntegrityReport,
    LedgerStats,
    RetentionPolicyList,
    RetentionPolicyOut,
    RetentionPolicyUpsert,
    RetentionResource,
    RetentionRunList,
    RetentionRunOut,
    RetentionStatus,
)
from app.schemas.auth import (
    AccessToken,
    LoginRequest,
    OrganizationOut,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPair,
    UserOut,
)
from app.schemas.documents import (
    DocumentList,
    DocumentListItem,
    DocumentOut,
    DocumentStatus,
    DocumentStatusOut,
    UploadResponse,
)
from app.schemas.training import (
    JobStatus,
    ModelStage,
    PromoteModelRequest,
    RegisteredModelDetail,
    RegisteredModelList,
    RegisteredModelSummary,
    RerankerPredictRequest,
    RerankerPredictResponse,
    ScoredPassage,
    TrainingJobDetail,
    TrainingJobList,
    TrainingJobMetricsOut,
    TrainingJobRequest,
    TrainingJobSummary,
)

__all__ = [
    "AccessToken",
    "AuditEventFilters",
    "AuditEventList",
    "AuditEventOut",
    "AuditOutcome",
    "DocumentList",
    "DocumentListItem",
    "DocumentOut",
    "DocumentStatus",
    "DocumentStatusOut",
    "IntegrityBreak",
    "IntegrityReport",
    "JobStatus",
    "LedgerStats",
    "LoginRequest",
    "ModelStage",
    "OrganizationOut",
    "PromoteModelRequest",
    "RefreshRequest",
    "RegisterRequest",
    "RegisterResponse",
    "RegisteredModelDetail",
    "RegisteredModelList",
    "RegisteredModelSummary",
    "RerankerPredictRequest",
    "RerankerPredictResponse",
    "RetentionPolicyList",
    "RetentionPolicyOut",
    "RetentionPolicyUpsert",
    "RetentionResource",
    "RetentionRunList",
    "RetentionRunOut",
    "RetentionStatus",
    "ScoredPassage",
    "TokenPair",
    "TrainingJobDetail",
    "TrainingJobList",
    "TrainingJobMetricsOut",
    "TrainingJobRequest",
    "TrainingJobSummary",
    "UploadResponse",
    "UserOut",
]
