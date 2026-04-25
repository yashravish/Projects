"""Documents API.

Routes:
    POST   /documents/upload     — multipart upload, schedules ingestion
    GET    /documents            — paginated list for the caller's org
    GET    /documents/{id}       — single document detail
    GET    /documents/{id}/status — light polling endpoint
    GET    /documents/{id}/pdf   — stream original bytes (auth required)
    DELETE /documents/{id}       — soft-delete
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.deps import CurrentUser, SessionDep
from app.logging_config import get_logger
from app.observability import audit_emitter
from app.schemas.documents import (
    DocumentList,
    DocumentListItem,
    DocumentOut,
    DocumentStatusOut,
    UploadResponse,
)
from app.services import document_service
from app.services.storage_service import (
    StorageError,
    build_default_scanner,
    get_storage,
)
from app.workers.ingestion_tasks import process_document

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger("api.documents")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    session: SessionDep,
    user: CurrentUser,
    file: UploadFile = File(..., description="PDF file (≤50MB, ≤500 pages)"),
    filename: str | None = Form(default=None),
) -> UploadResponse:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    storage = get_storage()
    scanner = build_default_scanner()
    safe_filename = filename or file.filename or "upload.pdf"

    try:
        doc, duplicate = await document_service.create_pending_document(
            session,
            organization_id=user.organization_id,
            user=user,
            filename=safe_filename,
            content_type=file.content_type,
            data=contents,
            storage=storage,
            scanner=scanner,
        )
    except document_service.DocumentServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    if not duplicate:
        # Schedule async ingestion. If broker is unreachable the worker will
        # never run; we still return 201 with status=pending so the user can
        # retry. The /documents/{id}/status endpoint is the source of truth.
        process_document.delay(str(user.organization_id), str(doc.id))

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="document.upload",
        resource_type="document",
        resource_id=doc.id,
        outcome="success",
        metadata={
            "filename": safe_filename,
            "byte_size": doc.byte_size,
            "sha256_prefix": doc.sha256[:16],
            "duplicate": duplicate,
        },
    )
    return UploadResponse(document_id=doc.id, status=doc.status, duplicate=duplicate)


@router.get("", response_model=DocumentList)
async def list_documents(
    session: SessionDep,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> DocumentList:
    items, total = await document_service.list_documents(
        session,
        organization_id=user.organization_id,
        page=page,
        page_size=page_size,
    )
    return DocumentList(
        items=[
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                page_count=doc.page_count,
                byte_size=doc.byte_size,
                status=doc.status,
                chunk_count=count,
                created_at=doc.created_at,
            )
            for doc, count in items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> DocumentOut:
    try:
        doc, count = await document_service.get_document(
            session,
            organization_id=user.organization_id,
            document_id=document_id,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc

    return DocumentOut(
        id=doc.id,
        organization_id=doc.organization_id,
        uploaded_by=doc.uploaded_by,
        filename=doc.filename,
        sha256=doc.sha256,
        page_count=doc.page_count,
        byte_size=doc.byte_size,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("/{document_id}/status", response_model=DocumentStatusOut)
async def document_status(
    document_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> DocumentStatusOut:
    try:
        doc, count = await document_service.get_document(
            session,
            organization_id=user.organization_id,
            document_id=document_id,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    return DocumentStatusOut(
        id=doc.id,
        status=doc.status,
        error_message=doc.error_message,
        page_count=doc.page_count,
        chunk_count=count,
        updated_at=doc.updated_at,
    )


@router.get("/{document_id}/pdf")
async def stream_pdf(
    document_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> StreamingResponse:
    try:
        doc, _ = await document_service.get_document(
            session,
            organization_id=user.organization_id,
            document_id=document_id,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc

    storage = get_storage()

    async def pdf_bytes() -> AsyncIterator[bytes]:
        try:
            async for chunk in storage.stream(doc.s3_key):
                yield chunk
        except StorageError as exc:
            raise HTTPException(
                status_code=502, detail=f"storage error: {exc}"
            ) from exc

    headers = {
        "Content-Disposition": f'inline; filename="{doc.filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(pdf_bytes(), media_type="application/pdf", headers=headers)


@router.delete("/{document_id}", response_class=Response, status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    user: CurrentUser,
) -> Response:
    storage = get_storage()
    try:
        await document_service.soft_delete_document(
            session,
            organization_id=user.organization_id,
            document_id=document_id,
            storage=storage,
        )
    except document_service.DocumentNotFoundError as exc:
        await audit_emitter.emit(
            session=session,
            organization_id=user.organization_id,
            actor_id=user.id,
            action="document.delete",
            resource_type="document",
            resource_id=document_id,
            outcome="denied",
            metadata={"reason": "not_found"},
        )
        raise HTTPException(status_code=404, detail="document not found") from exc
    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="document.delete",
        resource_type="document",
        resource_id=document_id,
        outcome="success",
    )
    return Response(status_code=204)
