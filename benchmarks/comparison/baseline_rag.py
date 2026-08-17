"""Naive embedding RAG baseline for comparison runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from common import (
    Document,
    RankedDocument,
    active_embedding_model_name,
    configure_benchmark_embedding_engine,
    embedding_engine,
)

MODE = "session_chunks_cosine_shared_embedding_engine"


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(query: Sequence[float], doc: Sequence[float], doc_norm: float) -> float:
    query_norm = _norm(query)
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return sum(q * d for q, d in zip(query, doc)) / (query_norm * doc_norm)


@dataclass(frozen=True)
class RagIndex:
    """Cosine top-k over benchmark chunks."""

    documents: tuple[Document, ...]
    embeddings: tuple[tuple[float, ...], ...]
    norms: tuple[float, ...]

    def search(self, query: str, top_k: int) -> list[RankedDocument]:
        """Return top-k vector matches."""
        query_embedding = embedding_engine.embed(query)
        ranked = self._rank(query_embedding)
        return ranked[:top_k]

    def _rank(self, query_embedding: Sequence[float]) -> list[RankedDocument]:
        pairs = [
            RankedDocument(doc, _cosine(query_embedding, emb, norm))
            for doc, emb, norm in zip(self.documents, self.embeddings, self.norms)
        ]
        pairs.sort(key=lambda item: (-item.score, item.document.doc_id))
        return pairs


def build_index(documents: Sequence[Document]) -> RagIndex:
    """Build a naive RAG index with the benchmark embedding model."""
    configure_benchmark_embedding_engine()
    frozen_docs = tuple(documents)
    texts = [doc.text for doc in frozen_docs]
    embeddings = tuple(tuple(item) for item in embedding_engine.embed_batch(texts))
    norms = tuple(_norm(item) for item in embeddings)
    return RagIndex(frozen_docs, embeddings, norms)


def embedding_model_name() -> str:
    """Expose the pinned benchmark embedding model for summaries."""
    return active_embedding_model_name()
