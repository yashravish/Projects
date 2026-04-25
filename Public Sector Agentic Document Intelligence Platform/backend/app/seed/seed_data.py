"""Idempotent seeding of the demo organization, admin user, and sample corpus.

Re-running this script is a no-op once the records exist; it never overwrites
existing data. Safe to invoke from the container entrypoint on every boot.

The sample corpus (three synthetic public-sector PDFs) is uploaded and
ingested in-process so the system has *real* data to demonstrate retrieval
on first boot. If `OPENAI_API_KEY` is set, embeddings are real; otherwise
the `LocalDeterministicEmbedder` is used.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.agents.llm_client import build_embedder
from app.config import get_settings
from app.db.models import Organization, User
from app.db.session import get_sessionmaker
from app.logging_config import configure_logging, get_logger
from app.security.passwords import hash_password
from app.seed.generate_sample_pdfs import build_sample_pdfs
from app.services.document_service import (
    DocumentServiceError,
    create_pending_document,
)
from app.services.ingestion_service import IngestionError, ingest_document
from app.services.storage_service import build_default_scanner, build_storage

log = get_logger("seed")


async def seed() -> None:
    settings = get_settings()
    sm = get_sessionmaker()

    async with sm() as session:
        org = await session.scalar(
            select(Organization).where(Organization.slug == settings.seed_org_slug)
        )
        created_org = False
        if org is None:
            org = Organization(name=settings.seed_org_name, slug=settings.seed_org_slug)
            session.add(org)
            await session.flush()
            created_org = True

        user = await session.scalar(
            select(User).where(User.email == settings.seed_admin_email.lower())
        )
        created_user = False
        if user is None:
            user = User(
                organization_id=org.id,
                email=settings.seed_admin_email.lower(),
                password_hash=hash_password(settings.seed_admin_password),
                role="admin",
                is_active=True,
            )
            session.add(user)
            created_user = True

        await session.commit()
        log.info(
            "seed.org_user",
            organization_id=str(org.id),
            user_email=user.email,
            org_created=created_org,
            user_created=created_user,
        )

    storage = build_storage()
    scanner = build_default_scanner()
    embedder = build_embedder()

    for sample in build_sample_pdfs():
        async with sm() as session:
            try:
                doc, duplicate = await create_pending_document(
                    session,
                    organization_id=org.id,
                    user=user,
                    filename=sample.filename,
                    content_type="application/pdf",
                    data=sample.bytes_,
                    storage=storage,
                    scanner=scanner,
                )
            except DocumentServiceError as exc:
                log.error("seed.upload_failed", filename=sample.filename, error=exc.message)
                continue

            if duplicate and doc.status == "ready":
                log.info(
                    "seed.document.already_ready",
                    filename=sample.filename,
                    document_id=str(doc.id),
                )
                continue

            try:
                chunk_count = await ingest_document(
                    session,
                    organization_id=org.id,
                    document_id=doc.id,
                    storage=storage,
                    embedder=embedder,
                )
            except IngestionError as exc:
                log.error(
                    "seed.ingest_failed",
                    filename=sample.filename,
                    document_id=str(doc.id),
                    error=str(exc),
                )
                continue

            log.info(
                "seed.document.ingested",
                filename=sample.filename,
                document_id=str(doc.id),
                chunks=chunk_count,
            )


def main() -> None:
    configure_logging()
    asyncio.run(seed())


if __name__ == "__main__":
    main()
