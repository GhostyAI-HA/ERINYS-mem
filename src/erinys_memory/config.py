"""ERINYS の設定値を保持する。"""

from dataclasses import dataclass, field
import os


def _parse_positive_int(raw: str, fallback: int) -> int:
    try:
        value = int(raw)
        return value if value > 0 else fallback
    except (ValueError, TypeError):
        return fallback


@dataclass
class ErinysConfig:
    db_path: str = os.environ.get("ERINYS_DB_PATH", "~/.erinys/memory.db")
    db_backup_on_init: bool = True
    db_max_size_mb: int = 500
    db_reader_pool_size: int = 4
    embedding_model: str = os.environ.get(
        "ERINYS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    embedding_dim: int = 384
    rrf_k: int = 60
    fts_weight: float = 0.4
    vec_weight: float = 0.6
    default_search_limit: int = 10
    max_search_limit: int = 500
    decay_lambda: float = 0.01
    reinforce_boost: float = 0.15
    prune_threshold: float = 0.1
    strength_cap: float = 2.0
    collider_sim_min: float = 0.65
    collider_sim_max: float = 0.90
    dream_max_collisions: int = 10
    max_content_length: int = 50_000
    max_title_length: int = 500
    enable_audit_log: bool = True
    prompt_retention_days: int = 30
    audit_retention_days: int = 90
    redact_secret_patterns: bool = True
    auto_distill_on_save: bool = os.environ.get("ERINYS_AUTO_DISTILL", "1") != "0"

    distill_model: str = os.environ.get("ERINYS_DISTILL_MODEL", "gemma4:e4b")
    distill_endpoint: str = os.environ.get(
        "ERINYS_DISTILL_ENDPOINT", "http://localhost:11434/api/generate"
    )
    distill_timeout: int = field(default_factory=lambda: _parse_positive_int(
        os.environ.get("ERINYS_DISTILL_TIMEOUT", "20"), 20
    ))
    distill_use_llm: bool = os.environ.get("ERINYS_DISTILL_USE_LLM", "1") != "0"

    def __post_init__(self) -> None:
        if self.distill_timeout <= 0:
            self.distill_timeout = 20
        if not self.distill_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
            import warnings
            warnings.warn(
                f"distill_endpoint '{self.distill_endpoint}' points outside localhost. "
                "ERINYS promises 'memory never leaves your machine'. "
                "Use a local endpoint to honor this contract.",
                UserWarning,
                stacklevel=2,
            )
