"""Export LaTeX tables and PNG figures from timing_comparison.json (+ optional Pareto sweep).

Run:
  python compare_cascade_timing.py --max-samples 200 --profile-ki
  python export_timing_latex.py
  python export_timing_latex.py --format png
  python export_timing_latex.py --format both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _ms(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "---"
    return f"{x:.{digits}f}"


def _pct_tex(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}\\%"


def _pct_str(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def _hardware_label(hw: dict) -> str:
    if hw.get("is_jetson") and hw.get("jetson_model"):
        return str(hw["jetson_model"])
    if hw.get("cuda_available") and hw.get("gpu_name"):
        return str(hw["gpu_name"])
    return f"{hw.get('device', 'unknown')} ({hw.get('platform', '')})"


def _pick_pareto_rows(pareto_path: Path, key_deadlines: list[float] | None) -> tuple[list[dict], int]:
    payload = json.loads(pareto_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    n_val = int(payload.get("n_val", len(rows)))
    if key_deadlines is None:
        deadlines = sorted(r["deadline_ms"] for r in rows)
        if len(deadlines) <= 5:
            return rows, n_val
        idx = [0, len(deadlines) // 4, len(deadlines) // 2, 3 * len(deadlines) // 4, -1]
        return [rows[i] for i in idx], n_val
    picked = []
    for d in key_deadlines:
        picked.append(min(rows, key=lambda r: abs(r["deadline_ms"] - d)))
    return picked, n_val


def _render_table_png(
    *,
    title: str,
    subtitle: str,
    col_labels: list[str],
    cell_rows: list[list[str]],
    footnote: str,
    output_path: Path,
    col_widths: list[float] | None = None,
) -> None:
    """Publication-style matplotlib table → PNG."""
    n_rows = len(cell_rows)
    n_cols = len(col_labels)
    fig_w = max(8.0, 1.6 * n_cols + 2)
    fig_h = max(3.0, 0.55 * n_rows + 2.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.axis("off")

    table = ax.table(
        cellText=cell_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.12, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_widths[col])
        if row == 0:
            cell.set_facecolor("#1a365d")
            cell.set_text_props(color="white", weight="bold", fontsize=11)
            cell.set_height(0.12)
        else:
            cell.set_facecolor("#ebf8ff" if row % 2 == 0 else "#f7fafc")
            cell.set_height(0.10)
            if col == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.04

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=9, color="#4a5568", wrap=True)
    fig.text(0.5, 0.04, footnote, ha="center", va="bottom", fontsize=8, color="#718096", style="italic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_ki_wcet_png(rows: list[dict], hardware: str, output_path: Path) -> None:
    cell_rows = []
    for row in rows:
        avg = row.get("table_avg_ms") or row.get("live_profile_avg_ms")
        wcet = row.get("table_wcet_ms") or row.get("live_profile_wcet_ms")
        cell_rows.append([row["ki"], _ms(avg), _ms(wcet)])
    _render_table_png(
        title="Per-Classifier Execution Costs",
        subtitle=f"{hardware} · batch size = 1 · perf_counter + GPU sync · 100 timed forwards",
        col_labels=["Classifier", "C̄_i (ms)", "WCET_i (ms)"],
        cell_rows=cell_rows,
        footnote="Kdet: simulated stub, WCET = 10,000 ms (paper penalty). Re-profile on Jetson for final numbers.",
        output_path=output_path,
        col_widths=[0.28, 0.36, 0.36],
    )


def export_cascade_timing_png(report: dict, output_path: Path) -> None:
    hw = _hardware_label(report.get("hardware", {}))
    t = report["cascade_table"]
    l = report["cascade_live"]
    n = report["n_val"]
    expand = report["expand_cost_ms"]

    cell_rows = [
        ["Accuracy", _pct_str(t["accuracy"]), _pct_str(l["accuracy"])],
        ["Mean latency C̄ (ms)", _ms(t["latency_ms"]["mean"]), _ms(l["latency_ms"]["mean"])],
        ["p50 latency (ms)", _ms(t["latency_ms"]["p50"]), _ms(l["latency_ms"]["p50"])],
        ["p95 latency (ms)", _ms(t["latency_ms"]["p95"], 0), _ms(l["latency_ms"]["p95"], 0)],
        ["Kdet rate", _pct_str(t["kdet_rate"]), _pct_str(l["kdet_rate"])],
        ["EXPAND E[C] (ms)", _ms(expand), _ms(expand)],
    ]
    _render_table_png(
        title="End-to-End EXPAND Cascade Timing",
        subtitle=f"n = {n} val samples · {hw} · table = sum of WCET_i · live = sum of measured forwards",
        col_labels=["Metric", "Table WCET", "Live measured"],
        cell_rows=cell_rows,
        footnote="Accuracy unchanged between modes; latency accounting differs (WCET vs measured).",
        output_path=output_path,
        col_widths=[0.40, 0.30, 0.30],
    )


def export_deadline_feasibility_png(
    pareto_path: Path,
    output_path: Path,
    *,
    key_deadlines: list[float] | None = None,
) -> None:
    picked, n_val = _pick_pareto_rows(pareto_path, key_deadlines)
    cell_rows = []
    for r in picked:
        d = r["deadline_ms"]
        d_str = f"{d:,.0f}" if d >= 100 else _ms(d, 1)
        cell_rows.append([d_str, _pct_str(r["accuracy"]), _pct_str(r["deadline_feasible_rate"])])
    acc = picked[0]["accuracy"]
    _render_table_png(
        title="Schedulability vs System Deadline D",
        subtitle=f"n = {n_val} val samples · guard disabled · feasible rate = Pr[latency ≤ D]",
        col_labels=["D (ms)", "Accuracy", "Feasible rate"],
        cell_rows=cell_rows,
        footnote=f"Accuracy constant at {_pct_str(acc)} (routing fixed; only feasibility varies with D).",
        output_path=output_path,
        col_widths=[0.34, 0.33, 0.33],
    )


def export_timing_pngs(
    comparison_path: Path,
    out_dir: Path,
    pareto_path: Path | None,
    *,
    key_deadlines: list[float] | None = None,
) -> list[Path]:
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    hw = _hardware_label(report.get("hardware", {}))
    paths: list[Path] = []

    p1 = out_dir / "table_ki_wcet.png"
    export_ki_wcet_png(report.get("per_ki_table_vs_profile", []), hw, p1)
    paths.append(p1)

    p2 = out_dir / "table_cascade_timing.png"
    export_cascade_timing_png(report, p2)
    paths.append(p2)

    if pareto_path is not None and pareto_path.is_file():
        p3 = out_dir / "table_deadline_feasibility.png"
        export_deadline_feasibility_png(pareto_path, p3, key_deadlines=key_deadlines)
        paths.append(p3)

    return paths


def table_ki_wcet(rows: list[dict], hardware: str) -> str:
    """Table: per-Ki measured avg and WCET (batch size 1)."""
    lines = [
        "% --- Table: Per-classifier execution costs ---",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-classifier inference latency on "
        f"{hardware}. Batch size $=1$; WCET from 100 timed forwards with "
        "\\texttt{perf\\_counter} and GPU synchronization.}}",
        "\\label{tab:ki-wcet}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Classifier & $\\bar{C}_i$ (ms) & $\\mathrm{WCET}_i$ (ms) \\\\",
        "\\midrule",
    ]
    for row in rows:
        ki = row["ki"].replace("_", "\\_")
        avg = row.get("table_avg_ms") or row.get("live_profile_avg_ms")
        wcet = row.get("table_wcet_ms") or row.get("live_profile_wcet_ms")
        lines.append(f"{ki} & {_ms(avg)} & {_ms(wcet)} \\\\")
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{3}{l}{\\footnotesize $K_{\\mathrm{det}}$: simulated stub, "
            "$\\mathrm{WCET}=10{,}000$\\,ms (paper penalty).} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def table_cascade_timing(report: dict) -> str:
    """Table: end-to-end cascade — table WCET vs live measured."""
    hw = _hardware_label(report.get("hardware", {}))
    t = report["cascade_table"]
    l = report["cascade_live"]
    n = report["n_val"]
    expand = report["expand_cost_ms"]

    lines = [
        "% --- Table: End-to-end cascade timing ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{End-to-end EXPAND cascade on $n="
        f"{n}$ validation samples ({hw}). "
        "Table mode sums per-$K_i$ WCET; live mode sums measured forward latency.}}",
        "\\label{tab:cascade-timing}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & Table WCET & Live measured \\\\",
        "\\midrule",
        f"Accuracy & {_pct_tex(t['accuracy'])} & {_pct_tex(l['accuracy'])} \\\\",
        f"Mean latency $\\bar{{C}}$ (ms) & {_ms(t['latency_ms']['mean'])} & {_ms(l['latency_ms']['mean'])} \\\\",
        f"p50 latency (ms) & {_ms(t['latency_ms']['p50'])} & {_ms(l['latency_ms']['p50'])} \\\\",
        f"p95 latency (ms) & {_ms(t['latency_ms']['p95'], 0)} & {_ms(l['latency_ms']['p95'], 0)} \\\\",
        f"$K_{{\\mathrm{{det}}}}$ rate & {_pct_tex(t['kdet_rate'])} & {_pct_tex(l['kdet_rate'])} \\\\",
        "\\midrule",
        f"EXPAND $\\mathbb{{E}}[C]$ (ms) & \\multicolumn{{2}}{{c}}{{{_ms(expand)}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_deadline_feasibility(pareto_path: Path, key_deadlines: list[float] | None = None) -> str:
    """Table: accuracy and feasible rate at selected system deadlines D."""
    picked, n_val = _pick_pareto_rows(pareto_path, key_deadlines)

    lines = [
        "% --- Table: Deadline feasibility (guard off) ---",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Schedulability vs.\\ system deadline $D$ "
        f"($n={n_val}$ val samples, guard disabled). "
        "Feasible rate $= \\Pr[\\text{latency} \\leq D]$.}}",
        "\\label{tab:deadline-feasibility}",
        "\\begin{tabular}{rrr}",
        "\\toprule",
        "$D$ (ms) & Accuracy & Feasible rate \\\\",
        "\\midrule",
    ]
    for r in picked:
        d = r["deadline_ms"]
        d_str = f"{d:,.0f}" if d >= 100 else _ms(d, 1)
        lines.append(
            f"{d_str} & {_pct_tex(r['accuracy'])} & {_pct_tex(r['deadline_feasible_rate'])} \\\\"
        )
    acc = picked[0]["accuracy"]
    lines.extend(
        [
            "\\midrule",
            f"\\multicolumn{{3}}{{l}}{{\\footnotesize Accuracy constant at {_pct_tex(acc)} "
            "(routing unchanged; only feasibility varies).}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_document(
    comparison_path: Path,
    pareto_path: Path | None,
    *,
    key_deadlines: list[float] | None = None,
) -> str:
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    hw = _hardware_label(report.get("hardware", {}))

    parts = [
        "% Auto-generated by export_timing_latex.py — paste into paper .tex",
        f"% Source: {comparison_path.name}",
        f"% Hardware at generation time: {hw}",
        "%",
        "% UPDATE: Re-run compare_cascade_timing.py on NVIDIA Jetson before submission.",
        "",
        table_ki_wcet(report.get("per_ki_table_vs_profile", []), hw),
        table_cascade_timing(report),
    ]

    if pareto_path is not None and pareto_path.is_file():
        parts.append(table_deadline_feasibility(pareto_path, key_deadlines))

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX timing tables for paper")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("checkpoints/timing_comparison.json"),
    )
    parser.add_argument(
        "--pareto",
        type=Path,
        default=Path("checkpoints/cascade_pareto_sweep.json"),
        help="Optional Pareto sweep JSON for deadline feasibility table",
    )
    parser.add_argument("--no-pareto", action="store_true")
    parser.add_argument(
        "--deadlines",
        type=str,
        default="17,140,1195,10500",
        help="Comma-separated D (ms) for feasibility table",
    )
    parser.add_argument("--out", type=Path, default=Path("checkpoints/timing_tables.tex"))
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for table PNG outputs",
    )
    parser.add_argument(
        "--format",
        choices=("tex", "png", "both"),
        default="both",
        help="Export LaTeX, PNG, or both",
    )
    args = parser.parse_args()

    if not args.comparison.is_file():
        raise FileNotFoundError(
            f"{args.comparison} not found. Run: python compare_cascade_timing.py --profile-ki"
        )

    key_d = [float(x.strip()) for x in args.deadlines.split(",") if x.strip()]
    pareto = None if args.no_pareto else args.pareto

    if args.format in ("tex", "both"):
        tex = build_document(args.comparison, pareto, key_deadlines=key_d)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex, encoding="utf-8")
        print(f"Wrote {args.out.resolve()}")

    if args.format in ("png", "both"):
        png_paths = export_timing_pngs(
            args.comparison,
            args.png_dir,
            pareto,
            key_deadlines=key_d,
        )
        for p in png_paths:
            print(f"Wrote {p.resolve()}")


if __name__ == "__main__":
    main()
