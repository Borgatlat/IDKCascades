"""Compare table WCET (wcet_profile.json) vs live perf_counter cascade timing.

RTS-compliant live timing uses time.perf_counter() + torch.cuda.synchronize() on GPU.

Run on target hardware (e.g. NVIDIA Jetson):
  python compare_cascade_timing.py
  python compare_cascade_timing.py --max-samples 500 --profile-ki
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cascade.eval import ExecutorConfig, evaluate_val_set, load_eval_bundle, summarize_traces
from cascade.executor import load_wcet_profile
from training.profile import profile_ki_wcet
from utils.hardware import hardware_info


from utils.hardware import hardware_info


def profile_all_ki(
    processed_dir: Path,
    checkpoint_dir: Path,
    registry_path: Path,
    *,
    timed_batches: int = 100,
) -> list[dict]:
    """Refresh per-Ki avg/WCET with GPU-synchronized forwards (batch_size=1)."""
    rows: list[dict] = []
    for ki in [f"K{i}" for i in range(7)] + ["Kdet"]:
        row = profile_ki_wcet(
            ki,
            processed_dir,
            batch_size=1,
            warmup_batches=10,
            timed_batches=timed_batches,
            checkpoint_dir=checkpoint_dir,
            registry_path=registry_path,
        )
        rows.append(row)
        print(
            f"  {ki}: avg={row['avg_ms']:.2f} ms  WCET={row['wcet_ms']:.2f} ms  "
            f"({row['batches']} timed batches)"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Table WCET vs live cascade timing")
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/timing_comparison.json"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-ki",
        action="store_true",
        help="Re-profile each Ki forward pass before cascade eval",
    )
    parser.add_argument("--profile-batches", type=int, default=100)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    registry_path = args.registry.resolve()
    device_info = hardware_info()
    print(f"Hardware: {device_info}")

    table_wcet = load_wcet_profile(checkpoint_dir)

    ki_profile_rows: list[dict] = []
    if args.profile_ki:
        print("Per-Ki live profiling (batch_size=1, CUDA sync, trained weights) ...")
        ki_profile_rows = profile_all_ki(
            args.processed_dir,
            checkpoint_dir,
            registry_path,
            timed_batches=args.profile_batches,
        )
        wcet_path = checkpoint_dir / "wcet_profile.json"
        wcet_path.write_text(json.dumps(ki_profile_rows, indent=2), encoding="utf-8")
        print(f"Updated {wcet_path}")
        table_wcet = load_wcet_profile(checkpoint_dir)

    bundle = load_eval_bundle(
        plan_path=args.plan.resolve(),
        processed_dir=args.processed_dir,
        checkpoint_dir=checkpoint_dir,
        registry_path=args.registry.resolve(),
        seed=args.seed,
        max_samples=args.max_samples,
    )

    print(f"Evaluating {len(bundle.val_idx)} val samples (guard off) ...")

    config_table = ExecutorConfig(
        timing_mode="table",
        wcet_ms=bundle.wcet_ms,
        deadline_ms=None,
        deadline_guard=False,
        kdet_sleep=False,
    )
    traces_table = evaluate_val_set(
        bundle.plan,
        bundle.models,
        bundle.registry,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        bundle.val_idx,
        bundle.device,
        config_table,
    )
    summary_table = summarize_traces(traces_table)

    config_live = ExecutorConfig(
        timing_mode="live",
        wcet_ms=bundle.wcet_ms,
        deadline_ms=None,
        deadline_guard=False,
        kdet_sleep=False,
    )
    traces_live = evaluate_val_set(
        bundle.plan,
        bundle.models,
        bundle.registry,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        bundle.val_idx,
        bundle.device,
        config_live,
    )
    summary_live = summarize_traces(traces_live)

    # Per-Ki table WCET vs freshly profiled avg/WCET.
    ki_compare: list[dict] = []
    profile_by_ki = {r["ki"]: r for r in ki_profile_rows}
    for entry in json.loads((checkpoint_dir / "wcet_profile.json").read_text(encoding="utf-8")):
        if entry["ki"] == "Kdet":
            continue
        ki = entry["ki"]
        live_row = profile_by_ki.get(ki, {})
        ki_compare.append(
            {
                "ki": ki,
                "table_avg_ms": entry.get("avg_ms"),
                "table_wcet_ms": entry.get("wcet_ms"),
                "live_profile_avg_ms": live_row.get("avg_ms"),
                "live_profile_wcet_ms": live_row.get("wcet_ms"),
            }
        )

    accuracy_match = summary_table["accuracy"] == summary_live["accuracy"]

    report = {
        "hardware": device_info,
        "n_val": len(bundle.val_idx),
        "expand_cost_ms": bundle.plan.expand_cost_ms,
        "cascade_table": summary_table,
        "cascade_live": summary_live,
        "delta_mean_latency_ms": summary_live["latency_ms"]["mean"] - summary_table["latency_ms"]["mean"],
        "accuracy_unchanged": accuracy_match,
        "per_ki_table_vs_profile": ki_compare,
        "note": (
            "cascade_table uses wcet_profile.json per Ki; cascade_live sums measured "
            "perf_counter forwards. Re-run on NVIDIA Jetson for paper hardware numbers."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("---")
    print(f"Accuracy:  table={summary_table['accuracy']:.4f}  live={summary_live['accuracy']:.4f}  match={accuracy_match}")
    print(
        f"Mean lat:  table={summary_table['latency_ms']['mean']:.2f} ms  "
        f"live={summary_live['latency_ms']['mean']:.2f} ms  "
        f"delta={report['delta_mean_latency_ms']:+.2f} ms"
    )
    print(f"EXPAND E[C]: {bundle.plan.expand_cost_ms:.2f} ms")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
