#!/usr/bin/env python3
"""Current Valid Claim v0 fixture をread-only shadow評価する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erinys_memory.claim_shadow import (
    ShadowComparison,
    compare_fixture,
    load_fixtures,
    suite_report,
)

DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "current_valid_claim_v0.jsonl"
)


class CliOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixtures: Path
    details: bool
    pretty: bool


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return CliOptions.model_validate(vars(parser.parse_args()))


def _payload(
    comparisons: list[ShadowComparison],
    details: bool,
) -> tuple[dict[str, object], int]:
    report = suite_report(comparisons)
    payload: dict[str, object] = {"report": report.model_dump(mode="json")}
    if details:
        payload["comparisons"] = [
            comparison.model_dump(mode="json") for comparison in comparisons
        ]
    return payload, report.failed


def run(options: CliOptions) -> tuple[dict[str, object], int]:
    fixtures = load_fixtures(options.fixtures)
    comparisons = [compare_fixture(fixture) for fixture in fixtures]
    payload, failed = _payload(comparisons, options.details)
    return payload, 1 if failed else 0


def main() -> int:
    options = parse_args()
    payload, exit_code = run(options)
    indent = 2 if options.pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
