"""Document business logic — read paths, soft delete, status, listing.

The write side (upload acceptance) is in `document_service.create_pending_document`
because it owns the dedupe-by-sha-and-org rule. Heavy work (extract → chunk →
embed → persist) lives in `ingestion_service` and runs in the Celery worker.

Every method takes `organization_id` as a required keyword argument. The unit
test in `tests/unit/test_tenant_isolation.py` enforces this with AST inspection.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, User
from app.logging_config import get_logger
from app.security.tenant import apply_tenant_filter
from app.services.storage_service import (
    InvalidPDFError,
    Storage,
    StorageError,
    StoredObject,
    VirusScanner,
    hash_bytes,
    make_object_key,
    validate_pdf_bytes,
)

log = get_logger("document_service")


class DocumentNotFoundError(Exception):
    """Raised when a document does not exist for the supplied tenant."""


class DocumentServiceError(Exception):
    """Raised on validation / persistence failures with a user-safe message."""

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


async def create_pending_document(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    user: User,
    filename: str,
    content_type: str | None,
    data: bytes,
    storage: Storage,
    scanner: VirusScanner,
) -> tuple[Document, bool]:
    """Validate + store a freshly uploaded PDF and create a `Document` row.

    Returns `(document, duplicate)` where `duplicate=True` means we returned
    an existing record for the same `(org, sha256)`.
    """
    try:
        validate_pdf_bytes(data, content_type=content_type)
    except InvalidPDFError as exc:
        raise DocumentServiceError(exc.message, code=exc.code, status_code=400) from exc

    if not await scanner.scan(data):
        raise DocumentServiceError(
            "uploaded file failed virus scan", code="VIRUS_DETECTED", status_code=400
        )

    sha = hash_bytes(data)

    existing = await session.scalar(
        apply_tenant_filter(
            select(Document).where(Document.sha256 == sha),
            Document,
            organization_id,
        )
    )
    if existing is not None:
        log.info(
            "document.upload.duplicate",
            organization_id=str(organization_id),
            document_id=str(existing.id),
            sha256=sha[:12],
        )
        return existing, True

    key = make_object_key(
        organization_id=organization_id, sha256=sha, filename=filename
    )

    try:
        stored: StoredObject = await storage.put(key, data)
    except StorageError as exc:
        raise DocumentServiceError(
            f"storage backend rejected upload: {exc}",
            code="STORAGE_ERROR",
            status_code=502,
        ) from exc

    doc = Document(
        organization_id=organization_id,
        uploaded_by=user.id,
        filename=filename,
        s3_key=stored.key,
        sha256=stored.sha256,
        byte_size=stored.bytes_written,
        page_count=0,
        status="pending",
    )
    session.add(doc)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = await session.scalar(
            apply_tenant_filter(
                select(Document).where(Document.sha256 == sha),
                Document,
                organization_id,
            )
        )
        if existing is None:
            raise
        return existing, True

    await session.refresh(doc)
    log.info(
        "document.upload.created",
        organization_id=str(organization_id),
        document_id=str(doc.id),
        bytes=stored.bytes_written,
    )
    return doc, False


async def list_documents(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[tuple[Document, int]], int]:
    """Return `(items, total)` where each item is `(document, chunk_count)`.

    Soft-deleted documents are excluded.
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 200))

    base = apply_tenant_filter(select(Document), Document, organization_id).where(
        Document.deleted_at.is_(None)
    )

    total = await session.scalar(
        apply_tenant_filter(
            select(func.count()).select_from(Document),
            Document,
            organization_id,
        ).where(Document.deleted_at.is_(None))
    )

    rows = (
        await session.execute(
            base.order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    if not rows:
        return [], int(total or 0)

    counts_stmt = (
        select(Chunk.document_id, func.count())
        .where(Chunk.document_id.in_([r.id for r in rows]))
        .group_by(Chunk.document_id)
    )
    counts = {
        doc_id: int(n)
        for doc_id, n in (await session.execute(counts_stmt)).all()
    }
    items = [(doc, counts.get(doc.id, 0)) for doc in rows]
    return items, int(total or 0)


async def get_document(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
) -> tuple[Document, int]:
    doc = await session.scalar(
        apply_tenant_filter(
            select(Document).where(Document.id == document_id),
            Document,
            organization_id,
        )
    )
    if doc is None or doc.deleted_at is not None:
        raise DocumentNotFoundError(str(document_id))
    chunk_count = int(
        await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
        or 0
    )
    return doc, chunk_count


async def soft_delete_document(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
    storage: Storage,
) -> None:
    doc = await session.scalar(
        apply_tenant_filter(
            select(Document).where(Document.id == document_id),
            Document,
            organization_id,
        )
    )
    if doc is None or doc.deleted_at is not None:
        raise DocumentNotFoundError(str(document_id))
    doc.deleted_at = dt.datetime.now(dt.UTC)
    # Cascade chunk delete — the FK is ON DELETE CASCADE; we explicitly delete
    # chunks here so the foreign-key cascade fires reliably even though the
    # parent row is only soft-deleted.
    await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    await session.commit()
    try:
        await storage.delete(doc.s3_key)
    except StorageError as exc:
        log.warning(
            "document.delete.storage_failed",
            document_id=str(doc.id),
            error=str(exc),
        )
    log.info(
        "document.delete",
        organization_id=str(organization_id),
        document_id=str(doc.id),
    )
