"""Hybrid retrieval over the chunks corpus.

Two signals are scored independently and fused with Reciprocal Rank Fusion (RRF):

1. **BM25-style sparse**: Postgres FTS via the precomputed `chunks.tsv` (a STORED
   GENERATED tsvector indexed by GIN). Scored with `ts_rank_cd` over a
   `plainto_tsquery` of the user's question.

2. **Dense vector**: pgvector cosine distance against the per-chunk `embedding`
   column (ivfflat indexed). Distances are turned into similarities (1 - d).

The two ranked lists are fused via RRF with `k=60`, then the top-K survivors
are returned with their per-signal positions and scores so the agent can
reason about provenance (e.g. only-FTS hits are typically rare keywords).

All queries are tenant-scoped at the SQL level — no result can ever leave its
`organization_id`.  Soft-deleted documents and any documents not yet `ready`
are excluded so partially-ingested data never bleeds into answers.

The retriever is a regular service (not a singleton): callers pass an
`AsyncSession` and `Embedder`. This keeps it trivially testable and matches
the rest of the service layer.
"""
from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import String

from app.agents.llm_client import EMBEDDING_DIM, Embedder
from app.logging_config import get_logger

log = get_logger("retrieval.hybrid")

RRF_K = 60
"""Reciprocal Rank Fusion constant. 60 is the value from the Cormack et al.
paper and is well-tested across heterogeneous signals."""


@dataclasses.dataclass(frozen=True)
class RetrievalSignal:
    """Per-signal score for an individual chunk in a hybrid result."""

    rank: int  # 1-based rank in this signal's ordering; -1 if not present
    score: float  # raw signal score (FTS rank or cosine sim)


@dataclasses.dataclass(frozen=True)
class RetrievedChunk:
    """A chunk surfaced by the hybrid retriever, with full provenance.

    `fused_score` is the RRF score; the per-signal records let the agent and
    the audit trail explain *why* a chunk was retrieved.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    organization_id: uuid.UUID
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    chunk_index: int
    text: str
    fused_score: float
    bm25: RetrievalSignal
    vector: RetrievalSignal


@dataclasses.dataclass(frozen=True)
class RetrievalConfig:
    """Tunables for a single retrieval call. Persisted on QueryRun for replay."""

    top_k: int = 6
    candidate_k: int = 25  # fetched per signal before fusion
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    min_fused_score: float = 0.0  # post-fusion cutoff; 0 disables

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


# ---- SQL ----------------------------------------------------------------------
#
# Both queries:
# - join `documents` so we can return the filename and gate on status / soft-delete,
# - take an explicit `:org_id` bind so tenant isolation is enforced at SQL level,
# - use a CTE that ranks within the SELECT (not in Python) so the planner can use
#   the FTS / IVFFlat indexes.

_BM25_SQL = text("""
    SELECT
        c.id              AS chunk_id,
        c.document_id     AS document_id,
        d.filename        AS document_filename,
        c.organization_id AS organization_id,
        c.page_start      AS page_start,
        c.page_end        AS page_end,
        c.char_start      AS char_start,
        c.char_end        AS char_end,
        c.chunk_index     AS chunk_index,
        c.text            AS text,
        ts_rank_cd(c.tsv, q.tsq) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN LATERAL plainto_tsquery('english', :query_text) AS q(tsq)
    WHERE
        c.organization_id = :org_id
        AND d.organization_id = :org_id
        AND d.deleted_at IS NULL
        AND d.status = 'ready'
        AND c.tsv @@ q.tsq
    ORDER BY score DESC
    LIMIT :limit
""").bindparams(
    bindparam("org_id", type_=PG_UUID(as_uuid=True)),
    bindparam("query_text", type_=String()),
    bindparam("limit"),
)


_VECTOR_SQL = text("""
    SELECT
        c.id              AS chunk_id,
        c.document_id     AS document_id,
        d.filename        AS document_filename,
        c.organization_id AS organization_id,
        c.page_start      AS page_start,
        c.page_end        AS page_end,
        c.char_start      AS char_start,
        c.char_end        AS char_end,
        c.chunk_index     AS chunk_index,
        c.text            AS text,
        1.0 - (c.embedding <=> CAST(:qvec AS vector)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE
        c.organization_id = :org_id
        AND d.organization_id = :org_id
        AND d.deleted_at IS NULL
        AND d.status = 'ready'
        AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:qvec AS vector) ASC
    LIMIT :limit
""").bindparams(
    bindparam("org_id", type_=PG_UUID(as_uuid=True)),
    bindparam("qvec", type_=String()),
    bindparam("limit"),
)


# ---- Helpers ------------------------------------------------------------------


def _format_pgvector(vec: Sequence[float]) -> str:
    """pgvector wants `[0.1,0.2,...]` (no spaces, square brackets) as TEXT."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def _row_to_chunk(
    row: Mapping[str, Any],
    *,
    bm25: RetrievalSignal,
    vector: RetrievalSignal,
    fused: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        document_filename=row["document_filename"],
        organization_id=row["organization_id"],
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        fused_score=fused,
        bm25=bm25,
        vector=vector,
    )


# ---- Retriever ----------------------------------------------------------------


class HybridRetriever:
    """Combines BM25 (FTS) and dense (pgvector) retrieval with RRF fusion."""

    def __init__(self, *, session: AsyncSession, embedder: Embedder) -> None:
        self._session = session
        self._embedder = embedder
        if embedder.dimension != EMBEDDING_DIM:
            raise ValueError(
                f"Embedder dimension {embedder.dimension} != schema dimension {EMBEDDING_DIM}"
            )

    async def retrieve(
        self,
        *,
        organization_id: uuid.UUID,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedChunk]:
        cfg = config or RetrievalConfig()
        if not query.strip():
            return []

        bm25_rows = await self._bm25(organization_id=organization_id, query=query, cfg=cfg)
        vec_rows = await self._vector(organization_id=organization_id, query=query, cfg=cfg)

        fused = self._fuse(bm25_rows, vec_rows, cfg=cfg)

        log.info(
            "retrieval.hybrid.complete",
            organization_id=str(organization_id),
            bm25_hits=len(bm25_rows),
            vector_hits=len(vec_rows),
            fused_hits=len(fused),
            top_k=cfg.top_k,
        )
        return fused[: cfg.top_k]

    # -- signals ---------------------------------------------------------------

    async def _bm25(
        self,
        *,
        organization_id: uuid.UUID,
        query: str,
        cfg: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        result = await self._session.execute(
            _BM25_SQL,
            {
                "org_id": organization_id,
                "query_text": query,
                "limit": cfg.candidate_k,
            },
        )
        return [dict(m) for m in result.mappings().all()]

    async def _vector(
        self,
        *,
        organization_id: uuid.UUID,
        query: str,
        cfg: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        embedded = await self._embedder.embed([query])
        if not embedded:
            return []
        qvec = _format_pgvector(embedded[0])
        result = await self._session.execute(
            _VECTOR_SQL,
            {
                "org_id": organization_id,
                "qvec": qvec,
                "limit": cfg.candidate_k,
            },
        )
        return [dict(m) for m in result.mappings().all()]

    # -- fusion ----------------------------------------------------------------

    def _fuse(
        self,
        bm25_rows: list[dict[str, Any]],
        vec_rows: list[dict[str, Any]],
        *,
        cfg: RetrievalConfig,
    ) -> list[RetrievedChunk]:
        """RRF fuse two ranked lists keyed by chunk_id.

        score(d) = sum over signals s of weight_s / (k + rank_s(d))
        """
        bm25_index: dict[uuid.UUID, tuple[int, float, dict[str, Any]]] = {}
        for i, row in enumerate(bm25_rows):
            bm25_index[row["chunk_id"]] = (i + 1, float(row["score"]), row)
        vec_index: dict[uuid.UUID, tuple[int, float, dict[str, Any]]] = {}
        for i, row in enumerate(vec_rows):
            vec_index[row["chunk_id"]] = (i + 1, float(row["score"]), row)

        all_ids = set(bm25_index) | set(vec_index)
        scored: list[RetrievedChunk] = []
        for cid in all_ids:
            b = bm25_index.get(cid)
            v = vec_index.get(cid)
            fused = 0.0
            if b is not None:
                fused += cfg.bm25_weight / (RRF_K + b[0])
            if v is not None:
                fused += cfg.vector_weight / (RRF_K + v[0])
            if fused < cfg.min_fused_score:
                continue
            if b is not None:
                row = b[2]
            else:
                assert v is not None  # noqa: S101 — cid is in vector index
                row = v[2]
            assert row is not None  # noqa: S101 — set membership guarantee
            scored.append(
                _row_to_chunk(
                    row,
                    bm25=RetrievalSignal(rank=b[0], score=b[1]) if b else RetrievalSignal(-1, 0.0),
                    vector=RetrievalSignal(rank=v[0], score=v[1]) if v else RetrievalSignal(-1, 0.0),
                    fused=fused,
                )
            )
        scored.sort(key=lambda c: c.fused_score, reverse=True)
        return scored


__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalSignal",
    "RetrievedChunk",
    "RRF_K",
]
