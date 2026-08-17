"""Pure lexical BM25 baseline for comparison runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bm25_impl import BM25Okapi, tokenize
from common import Document, RankedDocument

MODE = "okapi_bm25_k1_1.5_b_0.75"


@dataclass(frozen=True)
class BM25Index:
    """BM25 index over fixed benchmark documents."""

    documents: tuple[Document, ...]
    model: BM25Okapi

    def search(self, query: str, top_k: int) -> list[RankedDocument]:
        """Return top-k lexical matches."""
        ranked = self.model.rank(query, top_k)
        return [
            RankedDocument(self.documents[item.index], item.score)
            for item in ranked
        ]


def build_index(documents: Sequence[Document]) -> BM25Index:
    """Build a BM25 index without FTS or vector dependencies."""
    frozen_docs = tuple(documents)
    tokenized = [tokenize(doc.text) for doc in frozen_docs]
    return BM25Index(frozen_docs, BM25Okapi(tokenized))
