"""Generate single publication PNG for scene-shift calibration maintenance (B + C)."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.scene_calibration import plot_scene_calibration_paper_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot combined B+C paper figure")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("checkpoints/scene_shift_baseline.json"),
    )
    parser.add_argument(
        "--recalibration-json",
        type=Path,
        default=Path("checkpoints/scene_recalibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/figures/scene_calibration_maintenance_paper.png"),
    )
    args = parser.parse_args()

    out = plot_scene_calibration_paper_figure(
        args.baseline_json.resolve(),
        args.recalibration_json.resolve(),
        args.output.resolve(),
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
