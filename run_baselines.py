"""Compare EXPAND cascade vs linear and always-Kdet baselines (publishable table + PNG).

Run:
  python run_baselines.py --max-samples 500
  python run_baselines.py --kdet-mode model --timing live
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cascade.eval import evaluate_val_set, hardware_info, load_eval_bundle, summarize_traces
from cascade.executor import CascadePlan, ExecutorConfig
from cascade.kdet import KDET_MODES, build_kdet_context


def baseline_plans(expand: CascadePlan) -> dict[str, CascadePlan]:
    """EXPAND vs fixed-order baselines sharing the same specialized sub-cascades."""
    return {
        "expand": expand,
        "linear_k0_k3": CascadePlan(
            initial_cascade=["K0", "K1", "K2", "K3", "Kdet"],
            specialized_cascades=dict(expand.specialized_cascades),
            expand_cost_ms=expand.expand_cost_ms,
        ),
        "always_kdet": CascadePlan(
            initial_cascade=["Kdet"],
            specialized_cascades={},
            expand_cost_ms=0.0,
        ),
    }


def plot_baselines(rows: list[dict], output_path: Path) -> None:
    names = [r["name"] for r in rows]
    acc = [r["summary"]["accuracy"] * 100 for r in rows]
    lat = [r["summary"]["latency_ms"]["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    colors = ["#2b6cb0", "#718096", "#c05621"]
    for i, (n, a, l) in enumerate(zip(names, acc, lat)):
        ax.scatter(l, a, s=120, color=colors[i % len(colors)], edgecolors="#1a202c", zorder=3)
        ax.annotate(n, (l, a), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(r"Mean latency $\bar{C}$ (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cascade Baselines: Accuracy vs Latency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EXPAND vs baseline cascades")
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/baseline_comparison.json"))
    parser.add_argument("--out-png", type=Path, default=Path("checkpoints/baseline_comparison.png"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing", choices=("table", "live"), default="table")
    parser.add_argument("--kdet-mode", choices=KDET_MODES, default="registry")
    args = parser.parse_args()

    bundle = load_eval_bundle(
        plan_path=args.plan.resolve(),
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
        wcet_ms=bundle.wcet_ms.get("Kdet"),
    )
    config = ExecutorConfig(
        timing_mode=args.timing,
        wcet_ms=bundle.wcet_ms,
        deadline_ms=None,
        deadline_guard=False,
        kdet=kdet,
    )

    hw = hardware_info()
    plans = baseline_plans(bundle.plan)
    results: list[dict] = []

    print(f"Hardware: {hw}")
    print(f"Kdet: {args.kdet_mode} ({kdet.describe()})")
    print(f"Val samples: {len(bundle.val_idx)} | timing={args.timing}\n")

    for name, plan in plans.items():
        traces = evaluate_val_set(
            plan,
            bundle.models,
            bundle.registry,
            bundle.mic,
            bundle.geo,
            bundle.metadata,
            bundle.val_idx,
            bundle.device,
            config,
        )
        summary = summarize_traces(traces)
        row = {
            "name": name,
            "initial_cascade": plan.initial_cascade,
            "summary": summary,
        }
        results.append(row)
        print(
            f"{name:14s}  acc={summary['accuracy']:.3f}  "
            f"kdet={summary['kdet_rate']:.3f}  "
            f"mean_lat={summary['latency_ms']['mean']:.1f} ms"
        )

    payload = {
        "hardware": hw,
        "kdet_mode": args.kdet_mode,
        "kdet_description": kdet.describe(),
        "timing_mode": args.timing,
        "n_val": len(bundle.val_idx),
        "baselines": results,
        "paper_notes": {
            "kdet_oracle": "Use only for EXPAND probability-table profiling (not runtime eval).",
            "kdet_registry": "Default publishable mode: ~94% Kdet accuracy from validation CM.",
            "kdet_model": "Best: real Kdet forward pass on target hardware.",
            "timing": "Re-run with --timing live on NVIDIA Jetson before submission.",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_baselines(results, args.out_png)
    print(f"\nWrote {args.out_json.resolve()}")
    print(f"Wrote {args.out_png.resolve()}")


if __name__ == "__main__":
    main()
