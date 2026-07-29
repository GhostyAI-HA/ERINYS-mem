"""Compatibility helpers for the supported Python versions."""

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10 CI
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of enum.StrEnum."""
