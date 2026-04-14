"""Embedding 生成と sqlite-vec 向けシリアライズを提供する。"""

from __future__ import annotations

import struct

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

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
