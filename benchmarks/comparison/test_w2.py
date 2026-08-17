"""Offline W2 contract tests; no Mem0 provider call is made."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common
from mem0_adapter import credential_gate
from run_mem0_smoke import SMOKE_LIMITS


class W2ContractTests(unittest.TestCase):
    def test_csv_has_top_k_for_existing_rows(self) -> None:
        path = HERE / "results" / "comparison.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("top_k", rows[0])
        self.assertTrue(all(row["top_k"] for row in rows))

    def test_raw_longmemeval_loader_preserves_gold_document(self) -> None:
        cases = common.load_cases("longmemeval", 1)
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0].gold_ids <= {doc.doc_id for doc in cases[0].documents})

    def test_mem0_gate_blocks_without_provider_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            decision = credential_gate()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.required_env, "OPENAI_API_KEY")

    def test_mem0_limits_are_exactly_bounded(self) -> None:
        self.assertEqual(SMOKE_LIMITS, {"locomo": 5, "convomem": 2})


if __name__ == "__main__":
    unittest.main()
