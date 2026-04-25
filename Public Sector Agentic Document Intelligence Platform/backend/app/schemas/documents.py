"""Pydantic DTOs for the documents API."""
from __future__ import annotations

import datetime as dt
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


DocumentStatus = Literal[
    "pending", "extracting", "chunking", "embedding", "ready", "failed"
]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    uploaded_by: UUID | None
    filename: str
    sha256: str
    page_count: int
    byte_size: int
    status: DocumentStatus
    error_message: str | None
    chunk_count: int = 0
    created_at: dt.datetime
    updated_at: dt.datetime


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    page_count: int
    byte_size: int
    status: DocumentStatus
    chunk_count: int = 0
    created_at: dt.datetime


class DocumentList(BaseModel):
    items: list[DocumentListItem]
    total: int
    page: int
    page_size: int


class UploadResponse(BaseModel):
    document_id: UUID
    status: DocumentStatus
    duplicate: bool = Field(
        default=False,
        description="True if this SHA256 already existed for this org; the existing document was returned.",
    )


class DocumentStatusOut(BaseModel):
    id: UUID
    status: DocumentStatus
    error_message: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    updated_at: dt.datetime
