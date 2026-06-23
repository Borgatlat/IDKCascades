"""Render classifier registry as a clean table (PNG + HTML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.classifier_registry import ClassifierRegistry


def fmt_confusion_matrix(cm: list[list[int]]) -> str:
    """Format confusion matrix as aligned rows for table cells."""
    return "\n".join("  ".join(f"{v:4d}" for v in row) for row in cm)


def fmt_allowed_next(allowed: dict[str, list[str]]) -> str:
    """Format DAG routing as outcome → next classifiers."""
    return "\n".join(f"{outcome} → {', '.join(next_ki)}" for outcome, next_ki in allowed.items())


def build_display_df(registry_path: Path) -> pd.DataFrame:
    registry = ClassifierRegistry.load(registry_path)
    df = registry.to_dataframe().sort_values("name").reset_index(drop=True)

    return pd.DataFrame(
        {
            "name": df["name"],
            "runtime_ms": df["runtime_ms"].round(2),
            "p_idk": df["p_idk"].round(4),
            "p_correct": df["p_correct"].round(4),
            "confusion_matrix": df["confusion_matrix"].apply(fmt_confusion_matrix),
            "allowed_next": df["allowed_next"].apply(fmt_allowed_next),
        }
    )


def plot_table_png(df: pd.DataFrame, output_path: Path) -> None:
    """Draw a publication-style table figure with matplotlib."""
    # Tune these together: bigger fonts need taller/wider figure + taller cells.
    body_font = 16
    header_font = 17
    mono_font = 14
    title_font = 22
    dpi = 300

    col_labels = ["Ki", "C (ms)", "P(IDK)", "P(correct)", "Confusion matrix", "allowed_next"]
    n_rows = len(df)
    fig_h = max(10.0, 2.4 * n_rows + 3.5)

    fig, ax = plt.subplots(figsize=(28, fig_h))
    ax.axis("off")

    # bbox=[left, bottom, width, height] in axes coords — top ~10% left for title
    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
        bbox=[0, 0, 1, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(body_font)

    widths = [0.06, 0.07, 0.07, 0.08, 0.28, 0.44]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(widths[col])
        if row == 0:
            cell.set_facecolor("#2c5282")
            cell.set_text_props(color="white", weight="bold", fontsize=header_font)
            cell.set_height(0.10)
        else:
            cell.set_facecolor("#f7fafc" if row % 2 == 0 else "#ffffff")
            cell.set_height(0.24)
            font = mono_font if col >= 4 else body_font
            cell.get_text().set_fontsize(font)
            if col >= 4:
                cell.get_text().set_fontfamily("monospace")

    fig.suptitle(
        "Hierarchical IDK Cascade — Classifier Registry",
        fontsize=title_font,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.93, left=0.01, right=0.99, bottom=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.4)
    plt.close(fig)


def write_table_html(df: pd.DataFrame, output_path: Path) -> None:
    """Write a styled HTML table (easy to open in a browser)."""
    cols = ["name", "runtime_ms", "p_idk", "p_correct", "confusion_matrix", "allowed_next"]
    headers = ["Ki", "C (ms)", "P(IDK)", "P(correct)", "Confusion matrix", "allowed_next"]

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if col in ("confusion_matrix", "allowed_next"):
                cells.append(f"<td><pre>{val}</pre></td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Classifier Registry</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 2rem; background: #f8fafc; color: #1a202c; }}
    h1 {{ color: #1a365d; font-size: 1.75rem; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; background: white;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); border-radius: 8px; overflow: hidden; }}
    th {{ background: #2c5282; color: white; padding: 14px 16px; text-align: left; font-size: 1.05rem; }}
    td {{ padding: 14px 16px; border-bottom: 1px solid #e2e8f0; vertical-align: top; font-size: 1rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:nth-child(even) td {{ background: #f7fafc; }}
    pre {{ margin: 0; font-family: Consolas, "Courier New", monospace; font-size: 0.95rem;
           line-height: 1.45; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Hierarchical IDK Cascade — Classifier Registry</h1>
  <table>
    <thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classifier registry as a clean table")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("checkpoints/classifier_registry.json"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("checkpoints/classifier_registry_table.png"),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("checkpoints/classifier_registry_table.html"),
    )
    args = parser.parse_args()

    df = build_display_df(args.registry)
    plot_table_png(df, args.png)
    write_table_html(df, args.html)

    print(f"Wrote {args.png}")
    print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
