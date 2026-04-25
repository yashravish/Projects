"""Ingestion pipeline: extract → chunk → embed → persist.

Runs inside the Celery worker via `app.workers.ingestion_tasks.process_document`.
The state machine is:

    pending → extracting → chunking → embedding → ready
                                                  └────────┐
                       └──── failed ◄───────────────────────┘ (any step)

Each transition writes to `Document.status` so the frontend can poll. On
failure, `error_message` is populated with a sanitized excerpt.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.llm_client import Embedder
from app.chunking import ChunkConfig, chunk_pages, extract_pdf
from app.db.models import Chunk as ChunkModel
from app.db.models import Document
from app.logging_config import get_logger
from app.security.tenant import apply_tenant_filter
from app.services.storage_service import (
    InvalidPDFError,
    Storage,
    validate_page_count,
)

log = get_logger("ingestion_service")


class IngestionError(Exception):
    """Wrapping error for any ingestion-pipeline failure."""


async def _set_status(
    session: AsyncSession,
    *,
    document: Document,
    status: str,
    error_message: str | None = None,
    page_count: int | None = None,
) -> None:
    document.status = status
    if error_message is not None:
        document.error_message = error_message[:1000]
    if page_count is not None:
        document.page_count = page_count
    document.updated_at = dt.datetime.now(dt.UTC)
    await session.commit()
    log.info(
        "ingestion.status",
        document_id=str(document.id),
        status=status,
        error=error_message[:120] if error_message else None,
    )


async def ingest_document(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
    storage: Storage,
    embedder: Embedder,
    chunk_config: ChunkConfig | None = None,
) -> int:
    """Run the full pipeline for a single document. Returns the chunk count.

    Idempotent: re-running on a `ready` document deletes its existing chunks
    and re-builds them. This is what `POST /documents/upload` of a duplicate
    *can* trigger if a re-process is requested by an admin (Stage 5 hook).
    """
    document = await session.scalar(
        apply_tenant_filter(
            select(Document).where(Document.id == document_id),
            Document,
            organization_id,
        )
    )
    if document is None:
        raise IngestionError(f"document not found: {document_id}")

    try:
        await _set_status(session, document=document, status="extracting")
        data = await storage.get(document.s3_key)
        pages = extract_pdf(data)
        try:
            validate_page_count(len(pages))
        except InvalidPDFError as exc:
            raise IngestionError(exc.message) from exc

        await _set_status(
            session,
            document=document,
            status="chunking",
            page_count=len(pages),
        )
        cfg = chunk_config or ChunkConfig()
        chunks = chunk_pages([(p.page_number, p.text) for p in pages], cfg=cfg)
        if not chunks:
            raise IngestionError("PDF produced no extractable text")

        await _set_status(session, document=document, status="embedding")
        vectors = await embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise IngestionError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        # Replace any prior chunks for this document (idempotent re-ingest).
        await session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document.id)
        )
        await session.flush()

        for chunk, vec in zip(chunks, vectors, strict=True):
            session.add(
                ChunkModel(
                    document_id=document.id,
                    organization_id=document.organization_id,
                    chunk_index=chunk.index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text_content=chunk.text,
                    embedding=vec,
                    token_count=chunk.token_estimate,
                )
            )
        await _set_status(
            session, document=document, status="ready", page_count=len(pages)
        )
        log.info(
            "ingestion.complete",
            document_id=str(document.id),
            pages=len(pages),
            chunks=len(chunks),
        )
        return len(chunks)
    except Exception as exc:
        await session.rollback()
        # Re-fetch and mark failed in a fresh tx so the failure is durable.
        document = await session.scalar(
            apply_tenant_filter(
                select(Document).where(Document.id == document_id),
                Document,
                organization_id,
            )
        )
        if document is not None:
            await _set_status(
                session,
                document=document,
                status="failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        log.exception(
            "ingestion.failed",
            document_id=str(document_id),
            error=str(exc),
        )
        raise IngestionError(str(exc)) from exc
