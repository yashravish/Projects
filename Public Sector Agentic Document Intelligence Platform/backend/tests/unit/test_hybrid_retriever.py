"""Hybrid retriever — DB-backed test against the real `chunks` table.

Seeds two organisations with overlapping vocabularies so the test exercises:
- BM25 scoring (FTS keyword hit),
- vector scoring (deterministic embedder produces stable ranks),
- RRF fusion (chunks present in both signals beat singletons),
- tenant isolation (org B's chunks never appear in org A's results).
"""
from __future__ import annotations

import uuid

import pytest

from tests.conftest import requires_postgres

pytestmark = [pytest.mark.asyncio, requires_postgres]


async def _seed_org_with_chunks(
    session, *, name: str, chunks: list[tuple[str, str]]
) -> uuid.UUID:
    """Insert one org + one document + N chunks. Returns organization_id.

    Each chunk tuple is `(filename, text)`. Embeddings are computed by the
    `LocalDeterministicEmbedder` so the test does not need an OPENAI_API_KEY.
    """
    from app.agents.llm_client import LocalDeterministicEmbedder
    from app.db.models import Chunk, Document, Organization

    org = Organization(name=name, slug=name.lower().replace(" ", "-"))
    session.add(org)
    await session.flush()

    embedder = LocalDeterministicEmbedder()
    by_filename: dict[str, Document] = {}
    for i, (filename, text) in enumerate(chunks):
        if filename not in by_filename:
            doc = Document(
                organization_id=org.id,
                filename=filename,
                s3_key=f"test/{org.id}/{filename}",
                sha256=f"deadbeef{i:056d}",
                page_count=1,
                byte_size=len(text),
                status="ready",
            )
            session.add(doc)
            await session.flush()
            by_filename[filename] = doc

    embeddings = await embedder.embed([t for _, t in chunks])
    for i, ((filename, text), vec) in enumerate(zip(chunks, embeddings, strict=True)):
        doc = by_filename[filename]
        session.add(
            Chunk(
                document_id=doc.id,
                organization_id=org.id,
                chunk_index=i,
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=len(text),
                text_content=text,
                embedding=vec,
                token_count=len(text.split()),
            )
        )
    await session.commit()
    return org.id


@pytest.mark.asyncio
async def test_hybrid_retrieves_relevant_chunks_with_provenance(db_session) -> None:
    from app.agents.llm_client import LocalDeterministicEmbedder
    from app.retrieval.hybrid import HybridRetriever, RetrievalConfig

    org_a = await _seed_org_with_chunks(
        db_session,
        name="Agency A",
        chunks=[
            (
                "grant.pdf",
                "The Resilient Communities grant supports rural broadband "
                "expansion across eligible jurisdictions.",
            ),
            (
                "grant.pdf",
                "Applicants must submit a concept paper before the full "
                "proposal is reviewed by the program officer.",
            ),
            (
                "policy.pdf",
                "Public records requests must be acknowledged within five "
                "business days of receipt.",
            ),
        ],
    )

    retriever = HybridRetriever(
        session=db_session, embedder=LocalDeterministicEmbedder()
    )
    chunks = await retriever.retrieve(
        organization_id=org_a,
        query="rural broadband grant eligibility",
        config=RetrievalConfig(top_k=3, candidate_k=10),
    )

    assert chunks, "expected at least one chunk"
    # Top chunk should be the grant text, not the unrelated policy memo.
    assert chunks[0].document_filename == "grant.pdf"
    # Provenance is populated.
    top = chunks[0]
    assert top.fused_score > 0
    assert top.bm25.rank >= 1 or top.vector.rank >= 1


@pytest.mark.asyncio
async def test_hybrid_retriever_isolates_tenants(db_session) -> None:
    """Org A's results never include chunks from Org B, even on identical queries."""
    from app.agents.llm_client import LocalDeterministicEmbedder
    from app.retrieval.hybrid import HybridRetriever, RetrievalConfig

    text_a = "Procurement cycle 2026 closes on March 31 with sealed bids."
    text_b = "The same procurement cycle 2026 closes on March 31 with sealed bids."
    org_a = await _seed_org_with_chunks(
        db_session, name="Agency A", chunks=[("a.pdf", text_a)]
    )
    org_b = await _seed_org_with_chunks(
        db_session, name="Agency B", chunks=[("b.pdf", text_b)]
    )
    assert org_a != org_b

    retriever = HybridRetriever(
        session=db_session, embedder=LocalDeterministicEmbedder()
    )
    a_results = await retriever.retrieve(
        organization_id=org_a,
        query="procurement cycle 2026 sealed bids",
        config=RetrievalConfig(top_k=10),
    )
    b_results = await retriever.retrieve(
        organization_id=org_b,
        query="procurement cycle 2026 sealed bids",
        config=RetrievalConfig(top_k=10),
    )

    assert {c.organization_id for c in a_results} == {org_a}
    assert {c.organization_id for c in b_results} == {org_b}
    assert {c.document_filename for c in a_results} == {"a.pdf"}
    assert {c.document_filename for c in b_results} == {"b.pdf"}


@pytest.mark.asyncio
async def test_hybrid_retriever_empty_query_returns_nothing(db_session) -> None:
    from app.agents.llm_client import LocalDeterministicEmbedder
    from app.retrieval.hybrid import HybridRetriever

    org = await _seed_org_with_chunks(
        db_session,
        name="Agency Z",
        chunks=[("z.pdf", "Some plain text content.")],
    )
    retriever = HybridRetriever(
        session=db_session, embedder=LocalDeterministicEmbedder()
    )
    out = await retriever.retrieve(organization_id=org, query="   ")
    assert out == []


@pytest.mark.asyncio
async def test_hybrid_retriever_excludes_non_ready_documents(db_session) -> None:
    """Documents in `pending` / `failed` / soft-deleted state must be invisible."""
    from app.agents.llm_client import LocalDeterministicEmbedder
    from app.db.models import Chunk, Document, Organization
    from app.retrieval.hybrid import HybridRetriever

    org = Organization(name="Agency Q", slug="agency-q")
    db_session.add(org)
    await db_session.flush()

    pending = Document(
        organization_id=org.id,
        filename="pending.pdf",
        s3_key=f"test/{org.id}/pending.pdf",
        sha256="cafe" * 16,
        page_count=1,
        byte_size=10,
        status="pending",
    )
    db_session.add(pending)
    await db_session.flush()

    embedder = LocalDeterministicEmbedder()
    [vec] = await embedder.embed(["budget appropriations"])
    db_session.add(
        Chunk(
            document_id=pending.id,
            organization_id=org.id,
            chunk_index=0,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=10,
            text_content="budget appropriations report 2026",
            embedding=vec,
            token_count=4,
        )
    )
    await db_session.commit()

    retriever = HybridRetriever(session=db_session, embedder=embedder)
    out = await retriever.retrieve(organization_id=org.id, query="budget appropriations")
    assert out == [], "non-ready documents must not be retrievable"
