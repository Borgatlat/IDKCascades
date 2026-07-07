"""Build h24 *sensor* inventory (within-scene nodes) — WIP drift prototype.

For multi-scene M3N-VC work (h24 vs h08 vs s31 …), use process_scenes.py and
docs/scene_detector_plan.md. Scene detection targets environment IDs, not rs*.

Run:
  python profile_scene_inventory.py
  python profile_scene_inventory.py --plot
  python profile_scene_inventory.py --deploy-sensors rs6 --calibration-sensors rs1,rs2,rs3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data import DEFAULT_H24_DIR, DEFAULT_OUTPUT_DIR
from utils.scene_inventory import (
    build_scene_inventory,
    inventory_to_summary_table,
    load_segment_table,
    recommend_scene_split,
    write_inventory_html,
    write_inventory_json,
)

DEFAULT_OUT_DIR = Path("checkpoints")

# Consistent colors for global vehicle labels in coverage plot.
LABEL_COLORS = {
    "gle350": "#3182ce",
    "cx30": "#38a169",
    "mustang": "#dd6b20",
    "miata": "#805ad5",
    "background": "#718096",
}


def _parse_sensor_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def print_console_summary(
    inventory: dict,
    split_info: dict,
    warnings: list[str],
) -> None:
    """Human-readable summary for terminal."""
    print("\n=== h24 Scene Inventory (Sub-step A) ===")
    print(f"Scene proxy: {inventory['scene_proxy']}")
    print(f"Load source: {inventory.get('load_source', '?')}")
    print(f"Segments: {inventory['n_segments']:,}  ({inventory['duration_hours_total']:.2f} h)")
    print(f"Sensors: {inventory['n_sensors']}  {inventory['sensor_ids']}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    print("\nPer-sensor summary:")
    for sid in inventory["sensor_ids"]:
        s = inventory["sensors"][sid]
        print(
            f"  {sid}: {s['n_segments']:,} segments ({s['duration_hours']:.2f} h) | "
            f"global={s['global_labels']} | intermediate={s['intermediate_labels']}"
        )

    rec = split_info["recommended_split"]
    print("\nRecommended split:")
    print(f"  Strategy: {rec.get('strategy', split_info.get('strategy'))}")
    print(f"  Calibration sensors:   {rec.get('calibration_sensors', [])}")
    print(f"  Deployment sensors:    {rec.get('deployment_sensors', [])}")
    feasible = rec.get("feasible_for_shift_experiment", False)
    print(f"  Feasible for shift experiment: {'YES' if feasible else 'NO'}")
    blockers = rec.get("blockers", [])
    if blockers:
        print(f"  Blockers: {blockers}")

    if split_info.get("ineligible_sensors"):
        print("\nIneligible sensors (min segments / missing SUV or COUPE):")
        for sid, reasons in split_info["ineligible_sensors"].items():
            print(f"  {sid}: {reasons}")


def plot_coverage(inventory: dict, output_path: Path) -> None:
    """Stacked bar: segments per sensor_id colored by global_label."""
    crosstab = inventory["crosstab_global_label"]
    sensor_ids = inventory["sensor_ids"]
    all_labels = sorted({lbl for row in crosstab.values() for lbl in row.keys()})

    data = np.zeros((len(sensor_ids), len(all_labels)))
    for i, sid in enumerate(sensor_ids):
        row = crosstab.get(sid, {})
        for j, lbl in enumerate(all_labels):
            data[i, j] = row.get(lbl, 0)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    bottom = np.zeros(len(sensor_ids))
    x = np.arange(len(sensor_ids))

    for j, lbl in enumerate(all_labels):
        counts = data[:, j]
        color = LABEL_COLORS.get(lbl, "#a0aec0")
        ax.bar(x, counts, bottom=bottom, label=lbl, color=color, edgecolor="white", linewidth=0.5)
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(sensor_ids)
    ax.set_xlabel("Deployment scene (sensor_id)")
    ax.set_ylabel("Segment count (2 s windows)")
    ax.set_title("h24 Scene Coverage: Segments per Sensor by Vehicle Class", fontweight="bold")
    ax.legend(title="global_label", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile h24 deployment scenes (sensor_id inventory)")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_H24_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strategy", default="leave_one_sensor_out")
    parser.add_argument("--min-segments", type=int, default=100)
    parser.add_argument("--calibration-sensors", type=str, default=None)
    parser.add_argument("--deploy-sensors", type=str, default=None)
    parser.add_argument("--force-rebuild-table", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Write scene_inventory_coverage.png")
    args = parser.parse_args()

    cal_sensors = _parse_sensor_list(args.calibration_sensors)
    dep_sensors = _parse_sensor_list(args.deploy_sensors)
    if (cal_sensors is None) != (dep_sensors is None):
        parser.error("Provide both --calibration-sensors and --deploy-sensors, or neither.")

    metadata, warnings = load_segment_table(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        force_rebuild=args.force_rebuild_table,
    )

    inventory = build_scene_inventory(metadata)
    split_info = recommend_scene_split(
        inventory,
        strategy=args.strategy,
        min_segments=args.min_segments,
        calibration_sensors=cal_sensors,
        deployment_sensors=dep_sensors,
    )

    payload = {
        **inventory,
        "warnings": warnings,
        "split_analysis": split_info,
        "recommended_split": split_info["recommended_split"],
    }

    out_dir = args.out_dir.resolve()
    json_path = out_dir / "scene_inventory.json"
    csv_path = out_dir / "scene_inventory_table.csv"
    html_path = out_dir / "scene_inventory_table.html"

    write_inventory_json(payload, json_path)

    summary_df = inventory_to_summary_table(inventory)
    summary_df.to_csv(csv_path, index=False)
    write_inventory_html(
        summary_df,
        html_path,
        title="h24 Scene Inventory — Sensor Coverage",
    )

    print_console_summary(inventory, split_info, warnings)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")

    if args.plot:
        png_path = out_dir / "scene_inventory_coverage.png"
        plot_coverage(inventory, png_path)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
