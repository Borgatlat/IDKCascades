"""Build baseline cascade summary table + figure from run_cascade.py output.

Run:
  python run_cascade.py --out checkpoints/cascade_eval_baseline.json
  python plot_cascade_baseline.py
  python plot_cascade_baseline.py --eval-json checkpoints/cascade_eval_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EVAL = Path("checkpoints/cascade_eval_baseline.json")
DEFAULT_THRESHOLDS = Path("checkpoints/threshold_calibration.json")
DEFAULT_OUT_DIR = Path("checkpoints/figures/cascade_baseline")


def _pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"


def _ms(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def load_payload(eval_path: Path, threshold_path: Path) -> tuple[dict, dict | None]:
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    threshold_payload = None
    if threshold_path.is_file():
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return eval_payload, threshold_payload


def build_cascade_metrics_table(eval_payload: dict) -> pd.DataFrame:
    """End-to-end cascade metrics (one row)."""
    summary = eval_payload["summary"]
    lat = summary["latency_ms"]
    plan_path = Path(eval_payload["plan_path"])
    expand_cost = None
    if plan_path.is_file():
        expand_cost = float(json.loads(plan_path.read_text(encoding="utf-8")).get("expand_cost_ms", 0))

    return pd.DataFrame(
        [
            {
                "label": "baseline_paper_thresholds",
                "n_val": summary["n"],
                "accuracy": summary["accuracy"],
                "accuracy_pct": _pct(summary["accuracy"]),
                "kdet_rate": summary["kdet_rate"],
                "kdet_rate_pct": _pct(summary["kdet_rate"]),
                "mean_latency_ms": lat["mean"],
                "p50_latency_ms": lat["p50"],
                "p95_latency_ms": lat["p95"],
                "max_latency_ms": lat["max"],
                "expand_expected_ms": expand_cost,
                "timing_mode": eval_payload["config"]["timing_mode"],
                "kdet_mode": eval_payload["config"]["kdet_mode"],
            }
        ]
    )


def build_ki_threshold_table(threshold_payload: dict | None, eval_payload: dict) -> pd.DataFrame:
    """Per-Ki H_i, P(IDK), and how often each Ki fired in the cascade."""
    fire_counts = eval_payload["summary"].get("ki_fire_counts", {})
    n_val = int(eval_payload["summary"]["n"])

    rows: list[dict] = []
    if threshold_payload:
        for entry in threshold_payload.get("classifiers", []):
            ki = entry["ki"]
            fires = int(fire_counts.get(ki, 0))
            rows.append(
                {
                    "ki": ki,
                    "H_i": float(entry["threshold_hi"]),
                    "required_precision": float(entry["required_precision"]),
                    "achieved_precision": float(entry["achieved_precision"]),
                    "p_idk_val": float(entry["p_idk"]),
                    "n_fired_cascade": fires,
                    "fire_rate_cascade": fires / n_val if n_val else 0.0,
                    "fire_rate_pct": _pct(fires / n_val if n_val else 0.0),
                }
            )
    else:
        for ki, fires in sorted(fire_counts.items()):
            rows.append(
                {
                    "ki": ki,
                    "H_i": np.nan,
                    "required_precision": np.nan,
                    "achieved_precision": np.nan,
                    "p_idk_val": np.nan,
                    "n_fired_cascade": int(fires),
                    "fire_rate_cascade": fires / n_val if n_val else 0.0,
                    "fire_rate_pct": _pct(fires / n_val if n_val else 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_branch_table(eval_payload: dict) -> pd.DataFrame:
    """Routing branch counts from specialized sub-cascades."""
    branch_counts = eval_payload["summary"].get("branch_counts", {})
    n_val = int(eval_payload["summary"]["n"])
    rows = [
        {
            "branch_key": key,
            "n_samples": int(count),
            "fraction": count / n_val if n_val else 0.0,
            "fraction_pct": _pct(count / n_val if n_val else 0.0),
        }
        for key, count in sorted(branch_counts.items())
    ]
    return pd.DataFrame(rows)


def write_tables(
    cascade_df: pd.DataFrame,
    ki_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    cascade_csv = out_dir / "cascade_baseline_metrics.csv"
    cascade_df.to_csv(cascade_csv, index=False)
    paths["cascade_csv"] = cascade_csv

    ki_csv = out_dir / "cascade_baseline_ki_thresholds.csv"
    ki_df.to_csv(ki_csv, index=False)
    paths["ki_csv"] = ki_csv

    branch_csv = out_dir / "cascade_baseline_branches.csv"
    branch_df.to_csv(branch_csv, index=False)
    paths["branch_csv"] = branch_csv

    html_path = out_dir / "cascade_baseline_summary.html"
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Cascade Baseline Summary</title>",
        "<style>body{font-family:sans-serif;margin:2em;} table{border-collapse:collapse;margin-bottom:2em;}",
        "th,td{border:1px solid #ccc;padding:8px 12px;} th{background:#1a365d;color:white;}</style>",
        "</head><body>",
        "<h1>EXPAND Cascade Baseline (Paper H<sub>i</sub> Thresholds)</h1>",
        "<h2>End-to-End Metrics</h2>",
        cascade_df.to_html(index=False, float_format="%.4f"),
        "<h2>Per-Ki Thresholds &amp; Fire Rates</h2>",
        ki_df.to_html(index=False, float_format="%.4f"),
        "<h2>Routing Branches</h2>",
        branch_df.to_html(index=False, float_format="%.4f"),
        "</body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    paths["html"] = html_path
    return paths


def plot_baseline_figure(
    eval_payload: dict,
    ki_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Three-panel baseline figure: metrics, Ki fire rates, H_i thresholds."""
    summary = eval_payload["summary"]
    hw = eval_payload.get("hardware", {})
    device_label = hw.get("gpu_name") or hw.get("device") or "unknown"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="white")

    # Panel A: headline cascade metrics
    ax = axes[0]
    metric_labels = ["Accuracy", r"$K_{\mathrm{det}}$ rate", r"Mean $\bar{C}$ (ms)"]
    metric_values = [
        summary["accuracy"] * 100,
        summary["kdet_rate"] * 100,
        summary["latency_ms"]["mean"],
    ]
    colors = ["#3182ce", "#805ad5", "#38a169"]
    bars = ax.bar(metric_labels, metric_values, color=colors, edgecolor="#1a202c", linewidth=0.6)
    ax.set_title("End-to-End Cascade Baseline", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val, label in zip(bars, metric_values, metric_labels):
        fmt = f"{val:.1f}%" if "rate" in label.lower() or label == "Accuracy" else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                ha="center", va="bottom", fontsize=9)

    # Panel B: Ki invocation rate in cascade
    ax = axes[1]
    ki_plot = ki_df[ki_df["n_fired_cascade"] > 0].copy()
    if len(ki_plot) == 0:
        ki_plot = ki_df.copy()
    x = np.arange(len(ki_plot))
    fire_pct = ki_plot["fire_rate_cascade"].values * 100
    bars = ax.bar(x, fire_pct, color="#2b6cb0", edgecolor="#1a202c", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ki_plot["ki"].tolist())
    ax.set_ylabel("Fire rate in cascade (%)")
    ax.set_title(r"$K_i$ Invocation Rate", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fire_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8)

    # Panel C: calibrated H_i (log-friendly for near-zero K3)
    ax = axes[2]
    ki_all = ki_df.sort_values("ki")
    x = np.arange(len(ki_all))
    hi_vals = ki_all["H_i"].values
    bars = ax.bar(x, hi_vals, color="#dd6b20", edgecolor="#1a202c", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ki_all["ki"].tolist())
    ax.set_ylabel(r"Threshold $H_i$")
    ax.set_title(r"Per-$K_i$ Calibrated $H_i$", fontweight="bold")
    ax.set_ylim(0, min(1.05, max(hi_vals) * 1.15 + 0.05))
    ax.grid(axis="y", alpha=0.3)
    for bar, val, ki in zip(bars, hi_vals, ki_all["ki"]):
        label = f"{val:.3f}" if val >= 0.01 else "~0"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=7, rotation=45)

    acc = cascade_df.iloc[0]["accuracy_pct"]
    n = int(summary["n"])
    fig.suptitle(
        f"Baseline: paper precision-calibrated thresholds · n={n:,} · acc={acc} · {device_label}",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics_table_png(cascade_df: pd.DataFrame, ki_df: pd.DataFrame, output_path: Path) -> None:
    """Publication-style table image for slides / team call."""
    row = cascade_df.iloc[0]
    cascade_rows = [
        ["Validation samples", f"{int(row['n_val']):,}"],
        ["End-to-end accuracy", row["accuracy_pct"]],
        [r"$K_{\mathrm{det}}$ rate", row["kdet_rate_pct"]],
        [r"Mean latency $\bar{C}$", f"{row['mean_latency_ms']:.2f} ms"],
        ["p50 latency", f"{row['p50_latency_ms']:.2f} ms"],
        ["p95 latency", f"{row['p95_latency_ms']:.2f} ms"],
        ["EXPAND E[C] (WCET model)", f"{row['expand_expected_ms']:.1f} ms"],
        ["Timing mode", str(row["timing_mode"])],
    ]

    ki_rows = []
    for _, r in ki_df.iterrows():
        hi = r["H_i"]
        hi_str = f"{hi:.4f}" if hi >= 0.01 else "~0"
        ki_rows.append([
            r["ki"],
            hi_str,
            f"{100 * r['p_idk_val']:.1f}%" if pd.notna(r["p_idk_val"]) else "—",
            r["fire_rate_pct"],
        ])

    fig_h = max(6.0, 0.35 * (len(cascade_rows) + len(ki_rows) + 4))
    fig, axes = plt.subplots(2, 1, figsize=(10, fig_h), facecolor="white",
                             gridspec_kw={"height_ratios": [1, 1.2]})

    for ax, title, col_labels, cell_rows in [
        (axes[0], "End-to-End Cascade Baseline", ["Metric", "Value"], cascade_rows),
        (axes[1], r"Per-$K_i$ Thresholds (paper calibration)", ["Ki", r"$H_i$", "P(IDK) val", "Fire rate"], ki_rows),
    ]:
        ax.axis("off")
        table = ax.table(
            cellText=cell_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#ebf8ff" if r % 2 == 0 else "#f7fafc")
        ax.set_title(title, fontweight="bold", pad=12)

    fig.suptitle("Cascade Threshold Optimization — Baseline Reference", fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cascade baseline tables and figures")
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.eval_json.is_file():
        raise FileNotFoundError(
            f"{args.eval_json} not found. Run:\n"
            f"  python run_cascade.py --out {args.eval_json}"
        )

    eval_payload, threshold_payload = load_payload(args.eval_json, args.thresholds)
    cascade_df = build_cascade_metrics_table(eval_payload)
    ki_df = build_ki_threshold_table(threshold_payload, eval_payload)
    branch_df = build_branch_table(eval_payload)

    table_paths = write_tables(cascade_df, ki_df, branch_df, args.out_dir)
    chart_path = args.out_dir / "cascade_baseline_overview.png"
    table_png_path = args.out_dir / "cascade_baseline_table.png"

    plot_baseline_figure(eval_payload, ki_df, cascade_df, chart_path)
    plot_metrics_table_png(cascade_df, ki_df, table_png_path)

    row = cascade_df.iloc[0]
    print("\n=== Cascade Baseline Summary ===")
    print(f"  n={int(row['n_val']):,}  accuracy={row['accuracy_pct']}  "
          f"Kdet={row['kdet_rate_pct']}  C_bar={row['mean_latency_ms']:.2f} ms")
    print(f"\nWrote {table_paths['cascade_csv'].resolve()}")
    print(f"Wrote {table_paths['ki_csv'].resolve()}")
    print(f"Wrote {table_paths['branch_csv'].resolve()}")
    print(f"Wrote {table_paths['html'].resolve()}")
    print(f"Wrote {chart_path.resolve()}")
    print(f"Wrote {table_png_path.resolve()}")


if __name__ == "__main__":
    main()
