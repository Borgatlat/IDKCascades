"""One-shot Jetson (or any CUDA) profiling pipeline for paper-ready WCET numbers.

Run ON the Jetson (SSH or local terminal), not on a CPU-only Windows laptop:

  bash scripts/run_jetson_profile.sh
  python profile_jetson.py --max-samples 500

Steps:
  1. Per-Ki CUDA-sync profiling (K0–K6 + Kdet) with trained weights → wcet_profile.json
  2. Cascade eval: table WCET vs live perf_counter
  3. EXPAND vs linear vs always-Kdet baselines (live timing)
  4. Optional Pareto deadline sweep (guard off, fast)
  5. jetson_profile_report.json + updated timing_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cascade.eval import (
    default_deadline_sweep,
    evaluate_val_set,
    load_eval_bundle,
    summarize_traces,
    sweep_deadlines,
)
from cascade.executor import CascadePlan, ExecutorConfig, load_wcet_profile
from cascade.kdet import build_kdet_context
from run_baselines import baseline_plans
from training.profile import profile_ki_wcet
from utils.hardware import hardware_info, require_cuda_for_profiling


def profile_all_ki(
    processed_dir: Path,
    checkpoint_dir: Path,
    registry_path: Path,
    *,
    timed_batches: int,
) -> list[dict]:
    """Profile K0–K6 + Kdet with GPU sync and trained weights."""
    names = [f"K{i}" for i in range(7)] + ["Kdet"]
    rows: list[dict] = []
    for ki in names:
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
            f"p95={row['p95_ms']:.2f} ms"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson/CUDA one-shot WCET + cascade profiling")
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out-report", type=Path, default=Path("checkpoints/jetson_profile_report.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--profile-batches", type=int, default=100)
    parser.add_argument("--kdet-mode", choices=("registry", "model"), default="model")
    parser.add_argument("--skip-pareto", action="store_true")
    parser.add_argument("--pareto-max-samples", type=int, default=200)
    args = parser.parse_args()

    require_cuda_for_profiling()
    hw = hardware_info()
    checkpoint_dir = args.checkpoint_dir.resolve()
    registry_path = args.registry.resolve()

    print("=" * 60)
    print("Jetson / CUDA profiling pipeline")
    print(json.dumps(hw, indent=2))
    if not hw.get("is_jetson"):
        print("WARNING: CUDA detected but not a Jetson board — numbers are for this GPU only.")
    print("=" * 60)

    # --- Step 1: per-Ki WCET ---
    print("\n[1/4] Per-Ki profiling (batch_size=1, torch.cuda.synchronize) ...")
    ki_rows = profile_all_ki(
        args.processed_dir,
        checkpoint_dir,
        registry_path,
        timed_batches=args.profile_batches,
    )
    wcet_path = checkpoint_dir / "wcet_profile.json"
    wcet_path.write_text(json.dumps(ki_rows, indent=2), encoding="utf-8")
    print(f"Wrote {wcet_path}")

    kdet_row = next((r for r in ki_rows if r["ki"] == "Kdet"), None)
    kdet_measured_wcet = float(kdet_row["wcet_ms"]) if kdet_row else None

    bundle = load_eval_bundle(
        plan_path=args.plan.resolve(),
        processed_dir=args.processed_dir,
        checkpoint_dir=checkpoint_dir,
        registry_path=registry_path,
        seed=42,
        max_samples=args.max_samples,
    )

    kdet = build_kdet_context(
        args.kdet_mode,
        metrics_path=checkpoint_dir / "Kdet_metrics.json",
        checkpoint_dir=checkpoint_dir,
        registry_path=registry_path,
        wcet_ms=kdet_measured_wcet or bundle.wcet_ms.get("Kdet"),
    )

    # Inject measured Kdet WCET into bundle wcet map for table-mode comparisons.
    if kdet_measured_wcet is not None:
        bundle.wcet_ms["Kdet"] = kdet_measured_wcet

    # --- Step 2: table vs live cascade ---
    print(f"\n[2/4] Cascade eval on {len(bundle.val_idx)} val samples ...")
    config_table = ExecutorConfig(
        timing_mode="table",
        wcet_ms=bundle.wcet_ms,
        deadline_guard=False,
        kdet=kdet,
    )
    config_live = ExecutorConfig(
        timing_mode="live",
        wcet_ms=bundle.wcet_ms,
        deadline_guard=False,
        kdet=kdet,
    )
    traces_table = evaluate_val_set(
        bundle.plan, bundle.models, bundle.registry,
        bundle.mic, bundle.geo, bundle.metadata, bundle.val_idx, bundle.device, config_table,
    )
    traces_live = evaluate_val_set(
        bundle.plan, bundle.models, bundle.registry,
        bundle.mic, bundle.geo, bundle.metadata, bundle.val_idx, bundle.device, config_live,
    )
    summary_table = summarize_traces(traces_table)
    summary_live = summarize_traces(traces_live)
    print(
        f"  EXPAND acc={summary_live['accuracy']:.3f}  "
        f"table_mean={summary_table['latency_ms']['mean']:.1f} ms  "
        f"live_mean={summary_live['latency_ms']['mean']:.1f} ms"
    )

    timing_comparison = {
        "hardware": hw,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "n_val": len(bundle.val_idx),
        "expand_cost_ms": bundle.plan.expand_cost_ms,
        "per_ki_profile": ki_rows,
        "cascade_table": summary_table,
        "cascade_live": summary_live,
        "delta_mean_latency_ms": summary_live["latency_ms"]["mean"] - summary_table["latency_ms"]["mean"],
        "kdet_mode": args.kdet_mode,
        "kdet_measured_wcet_ms": kdet_measured_wcet,
    }
    timing_path = checkpoint_dir / "timing_comparison.json"
    timing_path.write_text(json.dumps(timing_comparison, indent=2), encoding="utf-8")
    print(f"Wrote {timing_path}")

    # --- Step 3: baselines (live) ---
    print("\n[3/4] Baselines (live timing) ...")
    baseline_rows: list[dict] = []
    config_live_base = ExecutorConfig(
        timing_mode="live",
        wcet_ms=bundle.wcet_ms,
        deadline_guard=False,
        kdet=kdet,
    )
    for name, plan in baseline_plans(bundle.plan).items():
        traces = evaluate_val_set(
            plan, bundle.models, bundle.registry,
            bundle.mic, bundle.geo, bundle.metadata, bundle.val_idx, bundle.device, config_live_base,
        )
        summary = summarize_traces(traces)
        baseline_rows.append({"name": name, "initial_cascade": plan.initial_cascade, "summary": summary})
        print(
            f"  {name:14s} acc={summary['accuracy']:.3f}  "
            f"mean_lat={summary['latency_ms']['mean']:.1f} ms"
        )
    baseline_path = checkpoint_dir / "baseline_comparison_jetson.json"
    baseline_path.write_text(
        json.dumps({"hardware": hw, "timing_mode": "live", "kdet_mode": args.kdet_mode, "baselines": baseline_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {baseline_path}")

    # --- Step 4: optional Pareto (guard off, table WCET from fresh profile) ---
    pareto_payload = None
    if not args.skip_pareto:
        print(f"\n[4/4] Pareto sweep (guard off, n={args.pareto_max_samples}) ...")
        pareto_bundle = load_eval_bundle(
            plan_path=args.plan.resolve(),
            processed_dir=args.processed_dir,
            checkpoint_dir=checkpoint_dir,
            registry_path=registry_path,
            seed=42,
            max_samples=args.pareto_max_samples,
        )
        pareto_bundle.wcet_ms = load_wcet_profile(checkpoint_dir)
        if kdet_measured_wcet is not None:
            pareto_bundle.wcet_ms["Kdet"] = kdet_measured_wcet
        deadlines = default_deadline_sweep(pareto_bundle.wcet_ms, steps=20)
        rows = sweep_deadlines(
            pareto_bundle,
            deadlines,
            timing_mode="live",
            deadline_guard=False,
            kdet=kdet,
        )
        pareto_path = checkpoint_dir / "cascade_pareto_sweep_jetson.json"
        pareto_payload = {
            "hardware": hw,
            "timing_mode": "live",
            "deadline_guard": False,
            "kdet_mode": args.kdet_mode,
            "rows": rows,
        }
        pareto_path.write_text(json.dumps(pareto_payload, indent=2), encoding="utf-8")
        print(f"Wrote {pareto_path}")
    else:
        print("\n[4/4] Pareto skipped (--skip-pareto)")

    report = {
        "hardware": hw,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "wcet_profile_path": str(wcet_path),
        "timing_comparison_path": str(timing_path),
        "baseline_comparison_path": str(baseline_path),
        "kdet_measured_wcet_ms": kdet_measured_wcet,
        "expand_live": summary_live,
        "paper_caption": (
            f"WCET profiled on {hw.get('jetson_model') or hw.get('gpu_name', 'CUDA GPU')} "
            f"with batch_size=1, torch.cuda.synchronize(), time.perf_counter()."
        ),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone. Report: {args.out_report.resolve()}")

    # Regenerate LaTeX tables if export script exists.
    export_script = Path("export_timing_latex.py")
    if export_script.is_file():
        print("Regenerating LaTeX/PNG tables ...")
        subprocess.run([sys.executable, str(export_script), "--format", "both"], check=False)


if __name__ == "__main__":
    main()
