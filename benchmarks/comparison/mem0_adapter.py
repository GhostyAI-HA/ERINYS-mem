"""Cost-gated Mem0 OSS adapter used by the W2 smoke runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDER_PROVIDER = "openai"
DEFAULT_EMBEDDER_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS = 1536
CHARS_PER_TOKEN = 4
REQUIRED_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class SourceDocument(Protocol):
    """Minimum document contract consumed by Mem0System."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Mem0Gate:
    """Credential decision that is safe to serialize."""

    provider: str
    model: str
    required_env: str
    allowed: bool
    reason: str


class CostGateBlocked(RuntimeError):
    """Raised before Mem0 initialization when paid access is unavailable."""

    def __init__(self, decision: Mem0Gate) -> None:
        super().__init__(decision.reason)
        self.decision = decision


@dataclass
class LLMCallMeter:
    """Collect provider usage without retaining prompts or secrets."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def callback(
        self,
        _client: object,
        response: object,
        params: Mapping[str, object],
    ) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        self.input_tokens += _usage_value(usage, "prompt_tokens") or _estimate_input(params)
        self.output_tokens += _usage_value(usage, "completion_tokens")


def credential_gate() -> Mem0Gate:
    """Return whether the configured Mem0 LLM may be initialized."""
    provider = os.getenv("MEM0_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    model = os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)
    required_env = REQUIRED_ENV_BY_PROVIDER.get(provider, "OPENAI_API_KEY")
    allowed = bool(os.getenv(required_env))
    reason = "allowed" if allowed else f"missing_required_env:{required_env}"
    return Mem0Gate(provider, model, required_env, allowed, reason)


def settings_snapshot(storage_dir: Path) -> dict[str, object]:
    """Return non-secret Mem0 settings for summaries and cost reports."""
    gate = credential_gate()
    embedder = os.getenv("MEM0_EMBEDDER_PROVIDER", DEFAULT_EMBEDDER_PROVIDER)
    return {
        "package": "mem0ai",
        "llm_provider": gate.provider,
        "llm_model": gate.model,
        "required_env": gate.required_env,
        "required_env_present": gate.allowed,
        "embedder_provider": embedder,
        "embedder_model": os.getenv("MEM0_EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL),
        "vector_store_provider": "qdrant",
        "vector_store_mode": "local_oss",
        "storage_dir": str(storage_dir),
    }


class Mem0System:
    """Mem0 OSS adapter with source-document ID projection."""

    def __init__(self, user_id: str, storage_dir: Path, meter: LLMCallMeter) -> None:
        decision = credential_gate()
        if not decision.allowed:
            raise CostGateBlocked(decision)
        from mem0 import Memory

        self.user_id = user_id
        self.meter = meter
        self._source_by_memory_id: dict[str, str] = {}
        self.memory = Memory.from_config(_memory_config(storage_dir, meter))

    def ingest(self, documents: Sequence[SourceDocument]) -> None:
        """Add each source document and retain the returned memory IDs."""
        for document in documents:
            result = self.memory.add(
                [{"role": "user", "content": document.text}],
                user_id=self.user_id,
                metadata={"source_id": document.doc_id},
                infer=True,
            )
            for item in _result_items(result):
                memory_id = str(item.get("id", ""))
                if memory_id:
                    self._source_by_memory_id[memory_id] = document.doc_id

    def search(self, query: str, top_k: int) -> list[str]:
        """Search Mem0 and project results back to source document IDs."""
        result = self.memory.search(
            query,
            top_k=top_k,
            filters={"user_id": self.user_id},
        )
        ranked: list[str] = []
        for item in _result_items(result):
            source_id = self._source_id(item)
            if source_id and source_id not in ranked:
                ranked.append(source_id)
        return ranked

    def close(self) -> None:
        """Close a local vector-store client when the provider exposes it."""
        client = getattr(getattr(self.memory, "vector_store", None), "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _source_id(self, item: Mapping[str, object]) -> str:
        memory_id = str(item.get("id", ""))
        mapped = self._source_by_memory_id.get(memory_id)
        if mapped:
            return mapped
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            return str(metadata.get("source_id", ""))
        return ""


def _memory_config(storage_dir: Path, meter: LLMCallMeter) -> dict[str, object]:
    storage_dir.mkdir(parents=True, exist_ok=True)
    llm_provider = os.getenv("MEM0_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    llm_config = {"model": os.getenv("MEM0_LLM_MODEL", DEFAULT_LLM_MODEL)}
    if str(llm_config["model"]).startswith("gpt-5"):
        # gpt-5 family rejects non-default temperature; mem0's name-based
        # auto-detection misses "gpt-5-mini", so force the reasoning set.
        llm_config["is_reasoning_model"] = True
    callback = meter.callback
    llm_config["response_callback"] = callback
    embedder_provider = os.getenv("MEM0_EMBEDDER_PROVIDER", DEFAULT_EMBEDDER_PROVIDER)
    embedder_config = {"model": os.getenv("MEM0_EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL)}
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_w2",
                "embedding_model_dims": DEFAULT_EMBEDDING_DIMS,
                "path": str(storage_dir / "qdrant"),
            },
        },
        "llm": {"provider": llm_provider, "config": llm_config},
        "embedder": {"provider": embedder_provider, "config": embedder_config},
        "history_db_path": str(storage_dir / "history.db"),
    }


def _result_items(value: object) -> list[Mapping[str, object]]:
    raw = value.get("results", []) if isinstance(value, Mapping) else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _usage_value(usage: object, name: str) -> int:
    value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
    return int(value) if isinstance(value, (int, float)) else 0


def _estimate_input(params: Mapping[str, object]) -> int:
    messages = params.get("messages", [])
    if not isinstance(messages, list):
        return 0
    text = " ".join(
        str(item.get("content", ""))
        for item in messages
        if isinstance(item, Mapping)
    )
    return max(len(text) // CHARS_PER_TOKEN, 1)


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens without sending text to a provider."""
    return max(len(text) // CHARS_PER_TOKEN, 1)


def json_snapshot(value: object) -> str:
    """Serialize safe adapter state for command output."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
