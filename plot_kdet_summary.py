"""Render Kdet (deterministic fallback) metrics as a publication-style PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def load_kdet_data(metrics_path: Path, registry_path: Path) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    kdet = next(c for c in registry["classifiers"] if c["name"] == "Kdet")
    return {"metrics": metrics, "registry": kdet}


def plot_kdet_summary(data: dict, output_path: Path) -> None:
    metrics = data["metrics"]
    reg = data["registry"]
    best = metrics["best_metrics"]
    class_names = reg["class_names"]
    cm = np.array(best["confusion_matrix"], dtype=float)
    recalls = [best["per_class_recall"][c] for c in class_names]

    history = metrics.get("history", [])
    epochs = [h["epoch"] for h in history]
    val_f1 = [h["val_macro_f1_present"] for h in history]
    best_epoch = max(range(len(val_f1)), key=lambda i: val_f1[i]) + 1 if val_f1 else 0

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.35, wspace=0.28)

    # --- Panel A: summary stats ---
    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.axis("off")
    summary_lines = [
        ("Level", "deterministic (no IDK)"),
        ("P(IDK)", f"{reg['p_idk']:.4f}  (always commits)"),
        ("P(resolves)", "1.0000"),
        ("P(correct)", f"{reg['p_correct']:.4f}"),
        ("Macro-F1", f"{reg['macro_f1']:.4f}"),
        ("Best epoch", str(best_epoch)),
        ("Loss", reg.get("loss_key", metrics.get("loss_key", "—"))),
        ("Modality", reg.get("modality", "both")),
        ("Params", f"{reg['num_params']:,}"),
        (r"$\bar{C}$ (avg)", f"{reg['runtime_ms']:.2f} ms"),
        ("WCET", f"{reg['wcet_ms']:.2f} ms"),
        ("H_i", "N/A (no deferral)"),
    ]
    cell_text = [[k, v] for k, v in summary_lines]
    table = ax_stats.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        bbox=[0, 0.05, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a365d")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#ebf4ff")
            cell.set_width(0.42)
        else:
            cell.set_facecolor("#f7fafc")
    ax_stats.set_title("Kdet — Deterministic Fallback (K_det)", fontsize=14, fontweight="bold", pad=12)

    # --- Panel B: per-class recall ---
    ax_recall = fig.add_subplot(gs[0, 1])
    colors = ["#2b6cb0" if r >= 0.93 else "#c05621" for r in recalls]
    bars = ax_recall.bar(class_names, recalls, color=colors, edgecolor="#1a202c", linewidth=0.6)
    ax_recall.axhline(0.93, color="#718096", linestyle="--", linewidth=1, label="93% reference")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.set_ylabel("Per-class recall")
    ax_recall.set_title("Validation Recall by Base Class", fontsize=13, fontweight="bold")
    ax_recall.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, recalls):
        ax_recall.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_recall.legend(loc="lower right", fontsize=9)

    # --- Panel C: confusion matrix ---
    ax_cm = fig.add_subplot(gs[1, 0])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=35, ha="right")
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion Matrix (row-normalized)", fontsize=13, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.55 else "#1a202c"
            ax_cm.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Recall fraction")

    # --- Panel D: training curve ---
    ax_curve = fig.add_subplot(gs[1, 1])
    if epochs:
        ax_curve.plot(epochs, val_f1, "o-", color="#2b6cb0", linewidth=2, markersize=4, label="Val macro-F1")
        if best_epoch:
            ax_curve.axvline(best_epoch, color="#c05621", linestyle="--", linewidth=1.2, label=f"Best (ep {best_epoch})")
        ax_curve.set_xlabel("Epoch")
        ax_curve.set_ylabel("Val macro-F1 (present classes)")
        ax_curve.set_ylim(max(0.65, min(val_f1) - 0.05), min(1.0, max(val_f1) + 0.03))
        ax_curve.legend(loc="lower right", fontsize=9)
    ax_curve.set_title("Training Progress (40 epochs)", fontsize=13, fontweight="bold")
    ax_curve.grid(True, alpha=0.3)

    fig.suptitle(
        "Kdet Deterministic Classifier — Validation Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Kdet deterministic classifier summary PNG")
    parser.add_argument("--metrics", type=Path, default=Path("checkpoints/Kdet_metrics.json"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/Kdet_summary.png"))
    args = parser.parse_args()

    data = load_kdet_data(args.metrics, args.registry)
    plot_kdet_summary(data, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
