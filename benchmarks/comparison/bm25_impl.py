"""Minimal Okapi BM25 implementation for the comparison baseline.

The score for query term q_i in document D is:

  IDF(q_i) * f(q_i,D) * (k1 + 1)
  / (f(q_i,D) + k1 * (1 - b + b * |D| / avgdl))

with Robertson-Sparck Jones IDF:

  log(1 + (N - df(q_i) + 0.5) / (df(q_i) + 0.5))

Defaults follow the standard Okapi BM25 setting used by many IR baselines:
k1=1.5 and b=0.75. This file intentionally has no third-party dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize text for a plain lexical baseline."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class BM25Score:
    """One scored document index."""

    index: int
    score: float


class BM25Okapi:
    """Small deterministic BM25 scorer over pre-tokenized documents."""

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = tuple(tuple(doc) for doc in documents)
        self.k1 = k1
        self.b = b
        self.doc_lens = tuple(len(doc) for doc in self.documents)
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.term_freqs = tuple(Counter(doc) for doc in self.documents)
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_freqs: Counter[str] = Counter()
        for doc in self.documents:
            doc_freqs.update(set(doc))
        return {term: self._idf_value(df) for term, df in doc_freqs.items()}

    def _idf_value(self, doc_freq: int) -> float:
        total = len(self.documents)
        return math.log(1.0 + (total - doc_freq + 0.5) / (doc_freq + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """Compute BM25 score for one document."""
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        freqs = self.term_freqs[doc_index]
        for token in query_tokens:
            score += self._term_score(token, freqs[token], doc_len)
        return score

    def _term_score(self, token: str, freq: int, doc_len: int) -> float:
        if freq <= 0:
            return 0.0
        denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
        return self.idf.get(token, 0.0) * freq * (self.k1 + 1.0) / denom

    def rank(self, query: str, limit: int) -> list[BM25Score]:
        """Rank all documents for a query."""
        tokens = tokenize(query)
        scores = [BM25Score(i, self.score(tokens, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda item: (-item.score, item.index))
        return scores[:limit]
