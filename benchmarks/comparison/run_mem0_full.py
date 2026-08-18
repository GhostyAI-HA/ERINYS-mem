"""Full Mem0 measurement with a live cost governor.

Reuses the smoke runner's per-case flow but removes the smoke ceiling and adds:
- per-case streamed progress (cumulative tokens and USD from the shared meter),
- a COST_ALERT marker once the projected total first exceeds --alert-usd,
- a hard abort (partial summary, status=ABORTED_COST_CAP) at --cap-usd.

Rates default to gpt-4.1-mini list prices (USD per 1M tokens) and are CLI-
configurable so a rate change never requires a code edit.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import common
import run_mem0_smoke as smoke
from mem0_adapter import LLMCallMeter, credential_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Mem0 measurement")
    parser.add_argument("--bench", choices=("locomo", "convomem"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alert-usd", type=float, default=40.0)
    parser.add_argument("--cap-usd", type=float, default=48.0)
    parser.add_argument("--input-usd-per-1m", type=float, default=0.40)
    parser.add_argument("--output-usd-per-1m", type=float, default=1.60)
    return parser.parse_args()


def _cost_usd(meter: LLMCallMeter, args: argparse.Namespace) -> float:
    return (
        meter.input_tokens * args.input_usd_per_1m
        + meter.output_tokens * args.output_usd_per_1m
    ) / 1_000_000


def main() -> None:
    args = parse_args()
    gate = credential_gate()
    if not gate.allowed:
        raise SystemExit(f"credential gate: {gate.reason}")
    cases = common.load_cases(args.bench, args.limit)
    total = len(cases)
    meter = LLMCallMeter()
    results: list[dict[str, object]] = []
    status = "COMPLETED"
    alerted = False
    started = time.time()
    for index, case in enumerate(cases, start=1):
        results.append(smoke._run_case(case, meter))
        cost = _cost_usd(meter, args)
        projected = cost / index * total
        print(
            f"[mem0-full] case {index}/{total} cum_calls={meter.calls} "
            f"cum_in={meter.input_tokens} cum_out={meter.output_tokens} "
            f"cum_usd={cost:.2f} projected_usd={projected:.2f} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        if not alerted and projected > args.alert_usd:
            alerted = True
            print(
                f"COST_ALERT projected_usd={projected:.2f} exceeds "
                f"alert_usd={args.alert_usd:.2f} at case {index}/{total} "
                f"(cum_usd={cost:.2f})",
                flush=True,
            )
        if cost >= args.cap_usd:
            status = "ABORTED_COST_CAP"
            print(
                f"COST_CAP_ABORT cum_usd={cost:.2f} >= cap_usd={args.cap_usd:.2f} "
                f"at case {index}/{total} — stopping to cap spend",
                flush=True,
            )
            break
    summary = smoke._completed_summary(args.bench, cases[: len(results)], results, gate, meter)
    summary["status"] = status
    summary["run_kind"] = "full"
    summary["requested_cases"] = total
    summary["cost_governor"] = {
        "alert_usd": args.alert_usd,
        "cap_usd": args.cap_usd,
        "input_usd_per_1m": args.input_usd_per_1m,
        "output_usd_per_1m": args.output_usd_per_1m,
        "final_cost_usd": round(_cost_usd(meter, args), 4),
        "alerted": alerted,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    result_path, summary_path = smoke._write_outputs(f"{args.bench}_full", summary, results)
    print(json.dumps({k: summary[k] for k in ("status", "n", "overall", "cost_governor")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"results_jsonl={result_path}", flush=True)
    print(f"summary_json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
