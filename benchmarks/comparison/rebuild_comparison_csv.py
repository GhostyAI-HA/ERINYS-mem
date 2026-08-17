#!/usr/bin/env python3
"""Migrate the W1 comparison CSV to the W2 schema."""

from __future__ import annotations

import common


def main() -> None:
    restored, unknown = common.rebuild_comparison_csv()
    print(f"rows={restored}")
    print(f"top_k_unknown={unknown}")


if __name__ == "__main__":
    main()
