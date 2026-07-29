#!/usr/bin/env python3
"""audit_logのCurrent Valid Claim shadowレシートをread-only集計する。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erinys_memory._sqlite import sqlite3
from erinys_memory.claim_shadow_runtime import (
    LiveShadowReceipt,
    build_soak_report,
)

DEFAULT_DB_PATH = "~/.erinys/memory.db"
SHADOW_OPERATION = "claim_shadow"


class CliOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db: Path
    limit: int = Field(gt=0, le=100_000)
    pretty: bool


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ERINYS_DB_PATH", DEFAULT_DB_PATH)).expanduser(),
    )
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--pretty", action="store_true")
    return CliOptions.model_validate(vars(parser.parse_args()))


def _connect_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _receipt_rows(
    connection: sqlite3.Connection, limit: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT detail
        FROM audit_log
        WHERE operation = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        [SHADOW_OPERATION, limit],
    ).fetchall()


def _parse_receipts(
    rows: list[sqlite3.Row],
) -> tuple[list[LiveShadowReceipt], int]:
    receipts: list[LiveShadowReceipt] = []
    invalid = 0
    for row in rows:
        try:
            receipts.append(LiveShadowReceipt.model_validate_json(str(row["detail"])))
        except ValidationError:
            invalid += 1
    return receipts, invalid


def run(options: CliOptions) -> dict[str, object]:
    connection = _connect_readonly(options.db)
    try:
        rows = _receipt_rows(connection, options.limit)
    finally:
        connection.close()
    receipts, invalid = _parse_receipts(rows)
    report = build_soak_report(receipts, invalid)
    return {"report": report.model_dump(mode="json")}


def main() -> int:
    options = parse_args()
    indent = 2 if options.pretty else None
    print(json.dumps(run(options), ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
