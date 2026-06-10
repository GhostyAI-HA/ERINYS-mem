"""Embedding 生成と sqlite-vec 向けシリアライズを提供する。"""

from __future__ import annotations

from collections.abc import Iterable
import struct
from typing import Protocol

# 多言語対応モデル（日英クロスリンガル検索のため。2026-06-10移行、dim=384は旧bge-small-en-v1.5と同一）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


class EmbeddingResult(Protocol):
    def tolist(self) -> list[float]:
        """NumPy-like embedding vector を list に変換する。"""


class TextEmbeddingModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[EmbeddingResult]:
        """テキスト配列を embedding に変換する。"""


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model: TextEmbeddingModel = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        """単一テキストを embedding する。"""
        results = list(self._model.embed([text]))
        return results[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストをまとめて embedding する。"""
        return [result.tolist() for result in self._model.embed(texts)]


def serialize_f32(values: list[float]) -> bytes:
    """sqlite-vec が受け取る float32 BLOB に変換する。"""
    return struct.pack(f"<{len(values)}f", *values)
