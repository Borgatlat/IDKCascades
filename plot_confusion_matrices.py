"""Publication-style confusion matrix figures for Ki classifiers (K0–K6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


# --- Matplotlib defaults tuned for paper / slide export ---
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
    }
)

LEVEL_LABELS = {
    "intermediate": "Intermediate",
    "global": "Global",
    "specialized_suv": "Specialized (SUV)",
    "specialized_coupe": "Specialized (Coupe)",
}

# Soft blue sequential map — reads well in print and on projectors.
_CMAP = LinearSegmentedColormap.from_list(
    "paper_blues", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
)


def _format_class_name(name: str) -> str:
    """Map registry slugs to compact, readable axis labels."""
    mapping = {
        "gle350": "GLE 350",
        "cx30": "CX-30",
        "mustang": "Mustang",
        "miata": "MX-5",
        "background": "Background",
        "suv": "SUV",
        "coupe": "Coupe",
    }
    return mapping.get(name.lower(), name.replace("_", " ").title())


def load_ki_entries(registry_path: Path, names: list[str]) -> list[dict]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    by_name = {c["name"]: c for c in registry["classifiers"]}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"Missing classifiers in registry: {missing}")
    return [by_name[n] for n in names]


def _annotation_text(count: int, fraction: float, show_percent: bool) -> str:
    if show_percent:
        return f"{count}\n{fraction:.0%}"
    return str(count)


def plot_single_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    class_names: list[str],
    *,
    title: str,
    subtitle: str | None = None,
    show_colorbar: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate_percent: bool = True,
    cbar_label: str = "Row recall",
) -> None:
    """Draw one row-normalized confusion matrix on the given axes."""
    labels = [_format_class_name(c) for c in class_names]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm))

    im = ax.imshow(cm_norm, cmap=_CMAP, vmin=vmin, vmax=vmax, aspect="equal")

    n = cm.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", labelpad=6)
    ax.set_ylabel("True label", labelpad=6)

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=11, fontweight="bold", pad=10, linespacing=1.25)

    # White grid between cells — standard in ML papers for readability.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    fontsize = 8 if n <= 3 else 7
    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            frac = cm_norm[i, j]
            text_color = "white" if frac > 0.62 else "#1a202c"
            ax.text(
                j,
                i,
                _annotation_text(count, frac, annotate_percent),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                linespacing=1.15,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)


def _ki_subtitle(entry: dict, *, compact: bool = False) -> str:
    level = LEVEL_LABELS.get(entry.get("level", ""), entry.get("level", ""))
    acc = entry.get("p_correct", 0.0)
    f1 = entry.get("macro_f1", 0.0)
    pidk = entry.get("p_idk", 0.0)
    if compact:
        return f"Acc {acc:.3f} · F1 {f1:.3f} · P(IDK) {pidk:.3f}"
    return f"{level}  |  Acc {acc:.3f}  |  F1 {f1:.3f}  |  P(IDK) {pidk:.3f}"


def plot_ki_confusion_matrix(entry: dict, output_path: Path) -> None:
    """Standalone high-resolution PNG for one Ki classifier."""
    cm = np.array(entry["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2 + divmod(cm.shape[0], 3)[1], 4.2), facecolor="white")
    plot_single_confusion_matrix(
        ax,
        cm,
        entry["class_names"],
        title=f"{entry['name']} — Validation Confusion Matrix",
        subtitle=_ki_subtitle(entry),
        show_colorbar=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def plot_all_confusion_matrices(entries: list[dict], output_path: Path) -> None:
    """Multi-panel figure: K0–K6 in a 2×4 grid (last cell unused)."""
    n_panels = len(entries)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(17, 4.8 * n_rows), facecolor="white")
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.72, wspace=0.42)

    level_by_ki = {
        entry["name"]: LEVEL_LABELS.get(entry.get("level", ""), entry.get("level", ""))
        for entry in entries
    }

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        cm = np.array(entry["confusion_matrix"], dtype=float)
        level = level_by_ki[entry["name"]]
        plot_single_confusion_matrix(
            ax,
            cm,
            entry["class_names"],
            title=f"{entry['name']} ({level})",
            subtitle=_ki_subtitle(entry, compact=True),
            show_colorbar=False,
        )

    # Shared colorbar on the right of the figure.
    cax = fig.add_axes([0.915, 0.18, 0.012, 0.62])
    sm = mpl.cm.ScalarMappable(cmap=_CMAP, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Row-normalized recall", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Hierarchical IDK Cascade — Validation Confusion Matrices (K0–K6)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.93, left=0.05, right=0.90, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot publication-style confusion matrices for K0–K6")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("checkpoints/classifier_registry.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/figures"),
    )
    parser.add_argument(
        "--ki",
        nargs="*",
        default=[f"K{i}" for i in range(7)],
        help="Classifier names to plot (default: K0 … K6)",
    )
    args = parser.parse_args()

    entries = load_ki_entries(args.registry, args.ki)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = args.output_dir / "confusion_matrices_K0_K6.png"
    plot_all_confusion_matrices(entries, combined_path)
    print(f"Wrote {combined_path}")

    for entry in entries:
        out = args.output_dir / f"{entry['name']}_confusion_matrix.png"
        plot_ki_confusion_matrix(entry, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
