"""Execute synthesized IDK cascades on the validation set.

Examples:
  # Fast accuracy / latency eval (table WCET, no Kdet sleep):
  python run_cascade.py --max-samples 50

  # Deadline-feasibility sweep (live timing + guard at D=500ms):
  python run_cascade.py --timing live --deadline-ms 500 --kdet-sleep

  # Full val set, write JSON report:
  python run_cascade.py --out checkpoints/cascade_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade.eval import evaluate_val_set, hardware_info, load_eval_bundle, summarize_traces
from cascade.executor import ExecutorConfig
from cascade.kdet import KDET_MODES, build_kdet_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthesized IDK cascade on val set")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("checkpoints/synthesized_cascades.json"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timing",
        choices=("table", "live"),
        default="table",
        help="table=WCET from wcet_profile (fast, matches EXPAND); live=perf_counter per forward",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=None,
        help="System deadline D_system (ms). Enables schedulability guard when set.",
    )
    parser.add_argument(
        "--no-deadline-guard",
        action="store_true",
        help="Disable D_remain shortcut-to-Kdet even when --deadline-ms is set",
    )
    parser.add_argument(
        "--kdet-mode",
        choices=KDET_MODES,
        default="registry",
        help="oracle=profiling only; registry=CM sampler ~94%%; model=real Kdet forward",
    )
    parser.add_argument(
        "--kdet-sleep",
        action="store_true",
        help="Real Kdet WCET sleep (10s). Default off for bulk eval.",
    )
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"{plan_path} not found. Run: python cascade/optimizer_stub.py"
        )

    bundle = load_eval_bundle(
        plan_path=plan_path,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        seed=args.seed,
        max_samples=args.max_samples,
    )

    kdet = build_kdet_context(
        args.kdet_mode,
        metrics_path=args.checkpoint_dir / "Kdet_metrics.json",
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        sleep=args.kdet_sleep,
        wcet_ms=bundle.wcet_ms.get("Kdet"),
    )

    config = ExecutorConfig(
        timing_mode=args.timing,
        wcet_ms=bundle.wcet_ms,
        deadline_ms=args.deadline_ms,
        deadline_guard=not args.no_deadline_guard,
        kdet_sleep=args.kdet_sleep,
        kdet=kdet,
    )

    hw = hardware_info()
    print(f"Plan: {' -> '.join(bundle.plan.initial_cascade)}  (EXPAND cost {bundle.plan.expand_cost_ms:.1f} ms)")
    print(f"Evaluating {len(bundle.val_idx)} val samples | timing={config.timing_mode} | device={bundle.device}")
    print(f"Kdet mode: {args.kdet_mode} ({kdet.describe()})")
    if config.deadline_ms is not None:
        guard = "on" if config.deadline_guard else "off"
        print(f"Deadline D={config.deadline_ms:.1f} ms | guard={guard}")

    traces = evaluate_val_set(
        bundle.plan,
        bundle.models,
        bundle.registry,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        bundle.val_idx,
        bundle.device,
        config,
    )

    summary = summarize_traces(traces, deadline_ms=config.deadline_ms)

    print("---")
    print(f"Accuracy:     {summary['accuracy']:.3f}")
    print(f"Kdet rate:    {summary['kdet_rate']:.3f}")
    print(f"Mean latency: {summary['latency_ms']['mean']:.2f} ms  (EXPAND expected {bundle.plan.expand_cost_ms:.2f} ms)")
    if config.deadline_ms is not None:
        print(f"Feasible @ D: {summary['deadline_feasible_rate']:.3f}  overrides={summary['deadline_override_rate']:.3f}")

    if args.out is not None:
        payload = {
            "plan_path": str(plan_path),
            "hardware": hw,
            "config": {
                "timing_mode": config.timing_mode,
                "deadline_ms": config.deadline_ms,
                "deadline_guard": config.deadline_guard,
                "kdet_sleep": config.kdet_sleep,
                "kdet_mode": args.kdet_mode,
                "kdet_description": kdet.describe(),
            },
            "summary": summary,
            "samples": [
                {
                    "idx": int(bundle.val_idx[i]),
                    "true_label": t.true_label,
                    "prediction": t.prediction,
                    "correct": t.correct,
                    "fired_kis": t.fired_kis,
                    "hit_kdet": t.hit_kdet,
                    "deadline_override": t.deadline_override,
                    "latency_ms": t.latency_ms,
                    "branch_key": t.branch_key,
                }
                for i, t in enumerate(traces)
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
