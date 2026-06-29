"""VMG Provenance: lineage-complete な出自記録の単一定義元 (#151)。

各 observation の metadata["provenance"] に「誰が・どの認可で・どう派生して・
いつ」記録されたかを構造化して残す。save/batch_save/distill/collide/import/
backfill 全経路から同じ schema で書けるよう、server.py と distill.py の双方が
ここを import する(循環 import 回避 + 単一スキーマ)。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

PROVENANCE_SCHEMA = 1
VALID_DERIVED_VIA = frozenset(
    {"save", "batch_save", "distill", "collide", "import", "backfill",
     "supersede", "session_summary"}
)


def build_provenance(
    source: str,
    principal: str | None,
    derived_via: str,
    parents: list[int] | None,
    recorded_at: str | None = None,
) -> dict[str, object]:
    """provenance ブロックを組み立てる。

    principal(誰が) は引数 > env ERINYS_PRINCIPAL > "unknown"。
    derived_via は既知集合外なら "save" に丸める。
    recorded_at は backfill 時に元の created_at を渡せるよう上書き可。
    """
    via = derived_via if derived_via in VALID_DERIVED_VIA else "save"
    return {
        "schema": PROVENANCE_SCHEMA,
        "principal": principal or os.environ.get("ERINYS_PRINCIPAL") or "unknown",
        "source": source,
        "derived_via": via,
        "parents": [int(p) for p in (parents or [])],
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
    }
