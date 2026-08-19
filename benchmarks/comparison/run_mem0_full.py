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
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime

import common
import run_mem0_smoke as smoke
from mem0_adapter import LLMCallMeter, Mem0System, credential_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Mem0 measurement")
    parser.add_argument("--bench", choices=("locomo", "convomem"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alert-usd", type=float, default=40.0)
    parser.add_argument("--cap-usd", type=float, default=48.0)
    parser.add_argument("--input-usd-per-1m", type=float, default=0.40)
    parser.add_argument("--output-usd-per-1m", type=float, default=1.60)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-case results from the progress checkpoint and run only the rest",
    )
    return parser.parse_args()


def _load_progress(path, resume: bool) -> dict[str, dict[str, object]]:
    """Return checkpointed case results; a non-resume run starts clean."""
    if not resume:
        path.unlink(missing_ok=True)
        return {}
    prior: dict[str, dict[str, object]] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    prior[str(record["case_id"])] = record
    return prior


def _cost_usd(meter: LLMCallMeter, args: argparse.Namespace) -> float:
    return (
        meter.input_tokens * args.input_usd_per_1m
        + meter.output_tokens * args.output_usd_per_1m
    ) / 1_000_000


class _Governor:
    """Streamed cost/progress bookkeeping shared by both run modes."""

    def __init__(self, args: argparse.Namespace, meter: LLMCallMeter, planned_units: int) -> None:
        self.args = args
        self.meter = meter
        self.planned_units = max(planned_units, 1)
        self.done_units = 0
        self.alerted = False
        self.status = "COMPLETED"
        self.started = time.time()

    def checkpoint(self, label: str) -> bool:
        """Record one finished unit; return False when the cap forces a stop."""
        self.done_units += 1
        cost = _cost_usd(self.meter, self.args)
        projected = cost / self.done_units * self.planned_units
        print(
            f"[mem0-full] {label} cum_calls={self.meter.calls} "
            f"cum_in={self.meter.input_tokens} cum_out={self.meter.output_tokens} "
            f"cum_usd={cost:.2f} projected_usd={projected:.2f} "
            f"elapsed={time.time()-self.started:.0f}s",
            flush=True,
        )
        if not self.alerted and projected > self.args.alert_usd:
            self.alerted = True
            print(
                f"COST_ALERT projected_usd={projected:.2f} exceeds "
                f"alert_usd={self.args.alert_usd:.2f} at {label} (cum_usd={cost:.2f})",
                flush=True,
            )
        if cost >= self.args.cap_usd:
            self.status = "ABORTED_COST_CAP"
            print(
                f"COST_CAP_ABORT cum_usd={cost:.2f} >= cap_usd={self.args.cap_usd:.2f} "
                f"at {label} — stopping to cap spend",
                flush=True,
            )
            return False
        return True


def _append_progress(path, record: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_per_case(
    args: argparse.Namespace,
    cases: list,
    prior: dict[str, dict[str, object]],
    progress_path,
    meter: LLMCallMeter,
) -> tuple[list[dict[str, object]], _Governor]:
    governor = _Governor(args, meter, len([c for c in cases if str(c.case_id) not in prior]))
    results: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        stored = prior.get(str(case.case_id))
        if stored is not None:
            results.append(stored)
            continue
        # Every case starts from an empty store so smoke leftovers or a
        # previous crash mid-ingest cannot double-ingest documents.
        shutil.rmtree(smoke._case_storage_dir(case), ignore_errors=True)
        result = smoke._run_case(case, meter)
        results.append(result)
        _append_progress(progress_path, result)
        if not governor.checkpoint(f"case {index}/{len(cases)}"):
            break
    return results, governor


def _conversation_groups(cases: list) -> dict[str, list]:
    """Group cases by conversation; refuse when grouping would be unsafe."""
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(str(case.case_id).rsplit(":", 1)[0], []).append(case)
    flattened = [c.case_id for members in groups.values() for c in members]
    if flattened != [c.case_id for c in cases]:
        raise SystemExit("conversation grouping would reorder cases; grouped mode unsafe")
    for key, members in groups.items():
        signatures = {
            tuple((d.doc_id, len(d.text)) for d in c.documents) for c in members
        }
        if len(signatures) != 1:
            raise SystemExit(f"conversation {key} has divergent document sets; grouped mode unsafe")
    return groups


def _group_storage_dir(benchmark: str, conv_key: str):
    digest = hashlib.sha256(conv_key.encode()).hexdigest()[:16]
    return smoke.RESULTS_DIR / "mem0_local" / benchmark / f"conv_{digest}"


def _run_grouped(
    args: argparse.Namespace,
    cases: list,
    prior: dict[str, dict[str, object]],
    progress_path,
    meter: LLMCallMeter,
) -> tuple[list[dict[str, object]], _Governor]:
    """Ingest each conversation once and answer all its questions from that store.

    LoCoMo attaches ~200 questions to each of 10 conversations; per-case
    ingestion would re-extract the same conversation for every question
    (54,916 LLM calls instead of 272) at ~200x the cost, and would also differ
    from how the ERINYS/RAG/BM25 benches index (once per conversation).
    """
    groups = _conversation_groups(cases)
    pending = [
        (key, members)
        for key, members in groups.items()
        if not all(str(c.case_id) in prior for c in members)
    ]
    governor = _Governor(args, meter, len(pending))
    results: list[dict[str, object]] = []
    for conv_index, (conv_key, members) in enumerate(groups.items(), start=1):
        if all(str(c.case_id) in prior for c in members):
            results.extend(prior[str(c.case_id)] for c in members)
            continue
        storage_dir = _group_storage_dir(args.bench, conv_key)
        shutil.rmtree(storage_dir, ignore_errors=True)
        system = Mem0System(conv_key, storage_dir, meter)
        ingest_started = time.perf_counter()
        system.ingest(members[0].documents)
        ingest_s = round(time.perf_counter() - ingest_started, 3)
        for position, case in enumerate(members):
            search_started = time.perf_counter()
            ranked_ids = system.search(case.question, smoke.DEFAULT_TOP_K)
            search_ms = (time.perf_counter() - search_started) * 1000.0
            ranked = smoke._ranked_documents(case, ranked_ids)
            recall, details = common.evaluate_case(case, ranked, smoke.DEFAULT_TOP_K)
            record: dict[str, object] = {
                "case_id": case.case_id,
                "category": case.category,
                "recall": recall,
                "retrieved_ids": ranked_ids,
                "details": details,
                "ingest_wall_s": ingest_s if position == 0 else 0.0,
                "latency_ms": round(search_ms, 3),
                "conversation_id": conv_key,
                "ingest_shared": True,
            }
            results.append(record)
            _append_progress(progress_path, record)
        system.close()
        # Drop the instance before the next conversation constructs its own:
        # a still-referenced Memory keeps ~/.mem0/migrations_qdrant locked.
        del system
        gc.collect()
        if not governor.checkpoint(
            f"conversation {conv_index}/{len(groups)} ({len(results)} cases)"
        ):
            break
    return results, governor


def main() -> None:
    args = parse_args()
    gate = credential_gate()
    if not gate.allowed:
        raise SystemExit(f"credential gate: {gate.reason}")
    cases = common.load_cases(args.bench, args.limit)
    total = len(cases)
    progress_path = smoke.RESULTS_DIR / f"mem0_full_progress_{args.bench}.jsonl"
    prior = _load_progress(progress_path, args.resume)
    if prior:
        print(f"[mem0-full] resume: {len(prior)} checkpointed cases reused", flush=True)
    meter = LLMCallMeter()
    if args.bench == "locomo":
        results, governor = _run_grouped(args, cases, prior, progress_path, meter)
    else:
        results, governor = _run_per_case(args, cases, prior, progress_path, meter)
    status = governor.status
    alerted = governor.alerted
    summary = smoke._completed_summary(args.bench, cases[: len(results)], results, gate, meter)
    summary["ingest_mode"] = "conversation_shared" if args.bench == "locomo" else "per_case"
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
        "resumed_prior_cases": len(prior),
        "cost_scope": "segment_only" if prior else "full_run",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    result_path, summary_path = smoke._write_outputs(f"{args.bench}_full", summary, results)
    if status == "COMPLETED":
        common._append_csv(summary)
    print(json.dumps({k: summary[k] for k in ("status", "n", "overall", "cost_governor")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"results_jsonl={result_path}", flush=True)
    print(f"summary_json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
