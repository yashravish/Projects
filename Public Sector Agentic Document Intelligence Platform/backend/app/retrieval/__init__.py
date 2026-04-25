"""Hybrid retrieval — BM25 (Postgres FTS) + dense (pgvector) + RRF fusion."""
from app.retrieval.hybrid import (
    HybridRetriever,
    RetrievalConfig,
    RetrievalSignal,
    RetrievedChunk,
)

__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalSignal",
    "RetrievedChunk",
]
