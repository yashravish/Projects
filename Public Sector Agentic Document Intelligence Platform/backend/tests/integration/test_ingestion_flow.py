"""End-to-end ingestion against a real Postgres + local storage + local embedder.

Skipped when Postgres isn't reachable. Celery is bypassed — we call the
ingestion service directly so the test stays in-process.
"""
from __future__ import annotations

import io

import pytest
from sqlalchemy import select

from app.agents.llm_client import LocalDeterministicEmbedder
from app.db.models import Chunk, Document
from app.seed.generate_sample_pdfs import build_sample_pdfs
from app.services.document_service import (
    DocumentServiceError,
    create_pending_document,
    get_document,
    list_documents,
    soft_delete_document,
)
from app.services.ingestion_service import ingest_document
from app.services.storage_service import (
    LocalFilesystemStorage,
    NullVirusScanner,
)
from tests.conftest import requires_postgres


@requires_postgres
@pytest.mark.asyncio
async def test_full_ingestion_round_trip(db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from app.db.models import Organization, User
    from app.security.passwords import hash_password

    org = Organization(name="Ingest Test Org", slug="ingest-test")
    db_session.add(org)
    await db_session.flush()
    user = User(
        organization_id=org.id,
        email="ingest@test.gov",
        password_hash=hash_password("Strong!Pass2026"),
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()

    storage = LocalFilesystemStorage(root=str(tmp_path))
    scanner = NullVirusScanner()
    embedder = LocalDeterministicEmbedder()

    sample = build_sample_pdfs()[0]

    doc, duplicate = await create_pending_document(
        db_session,
        organization_id=org.id,
        user=user,
        filename=sample.filename,
        content_type="application/pdf",
        data=sample.bytes_,
        storage=storage,
        scanner=scanner,
    )
    assert not duplicate
    assert doc.status == "pending"

    chunk_count = await ingest_document(
        db_session,
        organization_id=org.id,
        document_id=doc.id,
        storage=storage,
        embedder=embedder,
    )
    assert chunk_count > 0

    fresh = await db_session.scalar(select(Document).where(Document.id == doc.id))
    assert fresh.status == "ready"
    assert fresh.page_count >= 1
    assert fresh.error_message is None

    chunks = (
        await db_session.execute(select(Chunk).where(Chunk.document_id == doc.id))
    ).scalars().all()
    assert len(chunks) == chunk_count
    assert all(c.organization_id == org.id for c in chunks)
    assert all(c.embedding is not None for c in chunks)
    assert all(len(list(c.embedding)) == 1536 for c in chunks)

    # List endpoint logic.
    items, total = await list_documents(db_session, organization_id=org.id)
    assert total == 1
    assert items[0][1] == chunk_count

    # Get endpoint logic.
    fetched, count = await get_document(
        db_session, organization_id=org.id, document_id=doc.id
    )
    assert fetched.id == doc.id
    assert count == chunk_count

    # Duplicate upload returns the existing record without re-ingesting.
    dup_doc, is_dup = await create_pending_document(
        db_session,
        organization_id=org.id,
        user=user,
        filename=sample.filename,
        content_type="application/pdf",
        data=sample.bytes_,
        storage=storage,
        scanner=scanner,
    )
    assert is_dup
    assert dup_doc.id == doc.id

    # Soft-delete clears chunks and excludes from listing.
    await soft_delete_document(
        db_session, organization_id=org.id, document_id=doc.id, storage=storage
    )
    _, total_after = await list_documents(db_session, organization_id=org.id)
    assert total_after == 0
    remaining_chunks = (
        await db_session.execute(select(Chunk).where(Chunk.document_id == doc.id))
    ).scalars().all()
    assert remaining_chunks == []


@requires_postgres
@pytest.mark.asyncio
async def test_create_pending_rejects_non_pdf(db_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from app.db.models import Organization, User
    from app.security.passwords import hash_password

    org = Organization(name="Reject Org", slug="reject-org")
    db_session.add(org)
    await db_session.flush()
    user = User(
        organization_id=org.id,
        email="reject@test.gov",
        password_hash=hash_password("Strong!Pass2026"),
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()

    storage = LocalFilesystemStorage(root=str(tmp_path))
    scanner = NullVirusScanner()

    with pytest.raises(DocumentServiceError) as exc:
        await create_pending_document(
            db_session,
            organization_id=org.id,
            user=user,
            filename="evil.pdf",
            content_type="application/pdf",
            data=b"GIF89a not a real pdf",
            storage=storage,
            scanner=scanner,
        )
    assert exc.value.code == "NOT_A_PDF"


@requires_postgres
@pytest.mark.asyncio
async def test_documents_endpoint_lists_uploaded(client, db_session, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """API path: register, upload (background ingestion patched), list."""
    monkeypatch.setenv("LOCAL_UPLOAD_DIR", str(tmp_path))
    # Reset the storage singleton so it picks up the new path.
    from app.services import storage_service

    storage_service.reset_storage_for_tests()

    # Patch the celery .delay so we don't talk to a broker.
    from app.api.v1 import documents as documents_routes

    delays: list[tuple[str, str]] = []
    monkeypatch.setattr(
        documents_routes.process_document,
        "delay",
        lambda org, doc_id: delays.append((org, doc_id)),
    )

    r = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "doc-tester@example.gov",
            "password": "AnalystPass!2026",
            "organization_name": "Doc Tester Org",
        },
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    sample = build_sample_pdfs()[0]
    upload = await client.post(
        "/api/v1/documents/upload",
        files={"file": (sample.filename, io.BytesIO(sample.bytes_), "application/pdf")},
        headers=auth,
    )
    assert upload.status_code == 201, upload.text
    payload = upload.json()
    assert payload["status"] == "pending"
    assert payload["duplicate"] is False
    assert len(delays) == 1

    listing = await client.get("/api/v1/documents", headers=auth)
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    assert body["items"][0]["filename"] == sample.filename
