"""強度減衰と prune 判定を提供する。"""

from __future__ import annotations

import math
from datetime import datetime, timezone

LAMBDA = 0.01
REINFORCE_BOOST = 0.15
PRUNE_THRESHOLD = 0.1


def current_strength(
    base_strength: float,
    created_at: datetime,
    last_accessed: datetime | None,
    access_count: int,
    now: datetime | None = None,
) -> float:
    """
    Ebbinghaus-inspired decay with reinforcement.

    effective_strength = reinforced_base_strength × e^(-λt)
    reinforced_base_strength = base_strength + (access_count × REINFORCE_BOOST)

    effective_strength は常に current_strength() で read 時に計算する。
    erinys_reinforce は access_count と last_accessed のみ更新し、
    base_strength は直接更新しない。
    """
    now = now or datetime.now(timezone.utc)
    reference_time = last_accessed or created_at
    elapsed_days = (now - reference_time).total_seconds() / 86400
    reinforced_base = min(base_strength + access_count * REINFORCE_BOOST, 2.0)
    return reinforced_base * math.exp(-LAMBDA * elapsed_days)


def should_prune(strength: float) -> bool:
    return strength < PRUNE_THRESHOLD
