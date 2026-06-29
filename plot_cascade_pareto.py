"""Sweep D_system and plot accuracy vs deadline-feasibility (Pareto-style).

Default sweep: Ki WCET region through 1.05 × Kdet WCET (~10.5 s).

Run:
  python plot_cascade_pareto.py                             # full val, full D range, guard on
  python plot_cascade_pareto.py --no-deadline-guard        # faster: 1 pass, post-hoc feasible rate
  python plot_cascade_pareto.py --max-samples 50          # smoke test
  python plot_cascade_pareto.py --deadline-max 500         # Ki-only zoom (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from cascade.eval import (
    default_deadline_sweep,
    hardware_info,
    load_eval_bundle,
    sweep_deadlines,
)
from cascade.kdet import KDET_MODES, build_kdet_context


def parse_deadlines(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _format_d_ms(d: float) -> str:
    if d >= 1000:
        return f"{d / 1000:.1f}s" if d % 1000 == 0 else f"{d:.0f} ms"
    return f"{d:.0f} ms"


def plot_pareto(rows: list[dict], output_path: Path, *, plan_cost_ms: float) -> None:
    """
    Guard OFF — two-panel figure:
      Left  — accuracy vs deadline_feasible_rate (Pareto trade curve)
      Right — mean latency vs D_system (log x when full range)
    """
    feasible = np.array([r["deadline_feasible_rate"] for r in rows])
    accuracy = np.array([r["accuracy"] for r in rows])
    deadlines = np.array([r["deadline_ms"] for r in rows])
    mean_lat = np.array([r["latency_ms"]["mean"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    fig.subplots_adjust(wspace=0.38)

    # --- Panel A: accuracy vs feasible rate ---
    ax = axes[0]
    order = np.argsort(feasible)
    ax.plot(
        feasible[order],
        accuracy[order],
        "o-",
        color="#2b6cb0",
        linewidth=2,
        markersize=6,
        label="EXPAND cascade",
    )
    # Key D values in a text box (avoids labels bleeding into panel B).
    key_idx = [0, len(deadlines) // 2, len(deadlines) - 1]
    note_lines = [
        f"D = {_format_d_ms(deadlines[i])}: acc={accuracy[i]:.3f}, feas={feasible[i]:.3f}"
        for i in key_idx
    ]
    ax.text(
        0.03,
        0.03,
        "\n".join(note_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.95),
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max(0.0, accuracy.min() - 0.05), min(1.02, accuracy.max() + 0.03))
    ax.set_xlabel("Deadline-feasible rate  (latency ≤ D)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Pareto: Accuracy vs Schedulability", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # --- Panel B: mean latency vs D ---
    ax2 = axes[1]
    sc = ax2.scatter(
        deadlines,
        mean_lat,
        c=feasible,
        cmap="viridis",
        s=60,
        edgecolors="#1a202c",
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    ax2.plot(deadlines, deadlines, "--", color="#a0aec0", linewidth=1, label="D = mean latency")
    ax2.axhline(plan_cost_ms, color="#c05621", linestyle=":", linewidth=1.2, label=f"EXPAND E[C] = {plan_cost_ms:.0f} ms")
    ax2.set_xlabel(r"System deadline $D$ (ms)")
    ax2.set_ylabel(r"Mean processing latency $\bar{C}$ (ms)")
    ax2.set_title("Latency vs Bounded Deadline", fontsize=13, fontweight="bold")
    if deadlines.max() > 500.0:
        ax2.set_xscale("log")
        ax2.set_xlim(max(deadlines.min(), 1.0), deadlines.max() * 1.1)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper left", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.06)
    cbar.set_label("Feasible rate")

    fig.suptitle(
        "IDK Cascade — Deadline Feasibility (guard off)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pareto_guard(rows: list[dict], output_path: Path, *, plan_cost_ms: float, kdet_wcet_ms: float = 10_000.0) -> None:
    """
    Guard ON — Kdet override story with readable transition at D ≈ Kdet WCET.

    The cliff at ~10 s is real: below Kdet WCET the guard shortcuts every sample to
    Kdet (override=1, feasible=0); at D ≥ Kdet WCET the natural cascade runs.
    """
    deadlines = np.array([r["deadline_ms"] for r in rows])
    accuracy = np.array([r["accuracy"] for r in rows])
    feasible = np.array([r["deadline_feasible_rate"] for r in rows])
    override = np.array([r["deadline_override_rate"] for r in rows])
    mean_lat = np.array([r["latency_ms"]["mean"] for r in rows])
    natural_lat = float(mean_lat[override < 0.5].mean()) if np.any(override < 0.5) else plan_cost_ms

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), facecolor="white")
    fig.subplots_adjust(wspace=0.34)

    x_lo = max(deadlines.min(), 1.0)
    x_hi = deadlines.max() * 1.08

    # --- Panel A: accuracy vs D ---
    ax = axes[0]
    ax.step(deadlines, accuracy, where="post", color="#2b6cb0", linewidth=2, label="Accuracy")
    ax.plot(deadlines, accuracy, "o", color="#2b6cb0", markersize=5, zorder=3)
    ax.axvline(kdet_wcet_ms, color="#718096", linestyle="--", linewidth=1.2, label=f"Kdet WCET ({kdet_wcet_ms / 1000:.0f}s)")
    ax.axvspan(x_lo, kdet_wcet_ms, alpha=0.06, color="#c05621")
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(max(0.0, accuracy.min() - 0.02), min(1.02, accuracy.max() + 0.02))
    ax.set_xlabel(r"System deadline $D$ (ms)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Accuracy vs System Deadline", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)

    # --- Panel B: schedulability (step curves + zoom inset on transition) ---
    ax2 = axes[1]
    ax2.fill_between(
        np.append(deadlines[deadlines <= kdet_wcet_ms], kdet_wcet_ms),
        0,
        1,
        alpha=0.07,
        color="#c05621",
        step="post",
    )
    ax2.step(deadlines, override, where="post", color="#c05621", linewidth=2.5, label="Deadline override rate")
    ax2.step(deadlines, feasible, where="post", color="#2b6cb0", linewidth=2.5, label="Feasible rate (latency ≤ D)")
    ax2.plot(deadlines, override, "s", color="#c05621", markersize=6, zorder=4)
    ax2.plot(deadlines, feasible, "o", color="#2b6cb0", markersize=6, zorder=4)
    ax2.axvline(kdet_wcet_ms, color="#718096", linestyle="--", linewidth=1.2, label=f"Kdet WCET ({kdet_wcet_ms / 1000:.0f}s)")
    ax2.axhline(natural_lat, color="#38a169", linestyle=":", linewidth=1.2, label=f"Natural mean lat ≈ {natural_lat:.0f} ms")
    ax2.set_xscale("log")
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_xlabel(r"System deadline $D$ (ms)")
    ax2.set_ylabel("Rate")
    ax2.set_title("Schedulability Guard vs Deadline", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.0, 0.55))

    # Zoom inset: linear x around Kdet WCET (log scale compresses the cliff at the right edge).
    zoom_lo = max(kdet_wcet_ms * 0.75, float(deadlines[-3]) if len(deadlines) >= 3 else kdet_wcet_ms * 0.8)
    zoom_hi = float(deadlines.max()) * 1.02
    axins = inset_axes(ax2, width="42%", height="48%", loc="lower right", borderpad=1.8)
    axins.step(deadlines, override, where="post", color="#c05621", linewidth=2)
    axins.step(deadlines, feasible, where="post", color="#2b6cb0", linewidth=2)
    axins.plot(deadlines, override, "s", color="#c05621", markersize=7)
    axins.plot(deadlines, feasible, "o", color="#2b6cb0", markersize=7)
    axins.axvline(kdet_wcet_ms, color="#718096", linestyle="--", linewidth=1)
    axins.set_xlim(zoom_lo, zoom_hi)
    axins.set_ylim(-0.05, 1.08)
    axins.set_xlabel(r"$D$ (ms)", fontsize=8)
    axins.set_ylabel("Rate", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.set_title("Zoom: Kdet transition", fontsize=8, fontweight="bold")
    axins.grid(True, alpha=0.35)
    if len(deadlines) >= 1 and feasible[-1] >= 0.99:
        axins.annotate(
            "natural cascade",
            xy=(deadlines[-1], feasible[-1]),
            xytext=(-10, -30),
            textcoords="offset points",
            fontsize=7,
            ha="right",
            color="#2b6cb0",
            arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=0.8),
        )
    mark_inset(ax2, axins, loc1=1, loc2=3, fc="none", ec="#718096", linestyle=":", linewidth=1)

    fig.suptitle(
        "IDK Cascade — Deadline Guard Active (guard on)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_override_curve(rows: list[dict], output_path: Path) -> None:
    """Auxiliary: deadline override rate vs D (schedulability guard activity)."""
    deadlines = [r["deadline_ms"] for r in rows]
    override = [r["deadline_override_rate"] for r in rows]
    feasible = [r["deadline_feasible_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(deadlines, override, "s-", color="#c05621", label="Deadline override rate")
    ax.plot(deadlines, feasible, "o-", color="#2b6cb0", label="Feasible rate")
    ax.set_xlabel(r"System deadline $D$ (ms)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Schedulability Guard vs Deadline", fontsize=13, fontweight="bold")
    if max(deadlines) > 500.0:
        ax.set_xscale("log")
        ax.set_xlim(max(min(deadlines), 1.0), max(deadlines) * 1.1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cascade Pareto curve over D_system")
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/cascade_pareto.png"))
    parser.add_argument("--json-out", type=Path, default=Path("checkpoints/cascade_pareto_sweep.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing", choices=("table", "live"), default="table")
    parser.add_argument(
        "--deadlines",
        type=str,
        default=None,
        help="Comma-separated D values (ms). Default: auto linspace from wcet_profile",
    )
    parser.add_argument(
        "--deadline-steps",
        type=int,
        default=26,
        help="Auto-sweep points: Ki region + tail through 1.05×Kdet WCET",
    )
    parser.add_argument(
        "--deadline-max",
        type=float,
        default=None,
        help="Cap auto-sweep max D (ms). Default: 1.05 × Kdet WCET",
    )
    parser.add_argument(
        "--no-deadline-guard",
        action="store_true",
        help="Disable schedulability shortcut (feasible curve from fixed latencies only)",
    )
    parser.add_argument("--kdet-sleep", action="store_true")
    parser.add_argument(
        "--kdet-mode",
        choices=KDET_MODES,
        default="registry",
        help="oracle=profiling only; registry=~94%% CM sampler; model=real Kdet",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        help="Replot from existing cascade_pareto_sweep.json (skip inference)",
    )
    args = parser.parse_args()

    if args.plot_only is not None:
        payload = json.loads(Path(args.plot_only).read_text(encoding="utf-8"))
        rows = payload["rows"]
        plan_cost = float(payload.get("expand_cost_ms", 0.0))
        guard_on = bool(payload.get("deadline_guard", False))
        out = args.out
        if guard_on:
            plot_pareto_guard(rows, out, plan_cost_ms=plan_cost)
        else:
            plot_pareto(rows, out, plan_cost_ms=plan_cost)
        plot_override_curve(rows, Path("checkpoints/cascade_guard_curve.png"))
        print(f"Replotted {out.resolve()}")
        return

    plan_path = args.plan.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"{plan_path} not found. Run: python cascade/optimizer_stub.py")

    bundle = load_eval_bundle(
        plan_path=plan_path,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if args.deadlines:
        deadlines = parse_deadlines(args.deadlines)
    else:
        deadlines = default_deadline_sweep(bundle.wcet_ms, steps=args.deadline_steps)
        if args.deadline_max is not None:
            deadlines = [d for d in deadlines if d <= args.deadline_max]
            if not deadlines:
                deadlines = [args.deadline_max]

    kdet = build_kdet_context(
        args.kdet_mode,
        metrics_path=args.checkpoint_dir / "Kdet_metrics.json",
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        sleep=args.kdet_sleep,
        wcet_ms=bundle.wcet_ms.get("Kdet"),
    )

    print(f"Plan: {' -> '.join(bundle.plan.initial_cascade)}  (EXPAND {bundle.plan.expand_cost_ms:.1f} ms)")
    print(f"Kdet mode: {args.kdet_mode} ({kdet.describe()})")
    print(f"Sweeping {len(deadlines)} deadlines on {len(bundle.val_idx)} val samples ...")

    rows = sweep_deadlines(
        bundle,
        deadlines,
        timing_mode=args.timing,
        deadline_guard=not args.no_deadline_guard,
        kdet_sleep=args.kdet_sleep,
        kdet=kdet,
    )

    guard_on = not args.no_deadline_guard
    if guard_on:
        plot_pareto_guard(rows, args.out, plan_cost_ms=bundle.plan.expand_cost_ms)
    else:
        plot_pareto(rows, args.out, plan_cost_ms=bundle.plan.expand_cost_ms)
    guard_suffix = Path("checkpoints/cascade_guard_curve.png")
    plot_override_curve(rows, guard_suffix)

    payload = {
        "plan_path": str(plan_path),
        "hardware": hardware_info(),
        "expand_cost_ms": bundle.plan.expand_cost_ms,
        "n_val": len(bundle.val_idx),
        "timing_mode": args.timing,
        "deadline_guard": not args.no_deadline_guard,
        "kdet_mode": args.kdet_mode,
        "kdet_description": kdet.describe(),
        "deadlines_ms": deadlines,
        "rows": rows,
        "paper_notes": {
            "timing_cpu_provisional": "Re-profile with --timing live on NVIDIA Jetson before submission.",
            "kdet_registry": "Default publishable mode; guard-on accuracy reflects ~94% Kdet, not oracle 100%.",
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.out.resolve()}")
    print(f"Wrote {guard_suffix.resolve()}")
    print(f"Wrote {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
