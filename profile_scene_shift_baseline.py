"""Sub-step B: scene-shift calibration drift baseline (fixed H_i, no recalibration).

Run:
  python profile_scene_shift_baseline.py --plot
  python profile_scene_shift_baseline.py --max-samples 500 --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cascade.eval import hardware_info, load_eval_bundle
from cascade.executor import ExecutorConfig
from cascade.kdet import KDET_MODES, build_kdet_context
from training.trainer import load_spectrogram_cache
from utils.scene_calibration import (
    build_scene_index_sets,
    compare_cascade_scenes,
    drift_detected,
    ki_drift_summary,
    load_scene_split,
    profile_all_kis,
    write_all_figures,
)

DEFAULT_FIG_DIR = Path("checkpoints/figures/scene_shift")


def _parse_sensor_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def print_console_summary(
    split: dict,
    drift_rows: list[dict],
    cascade_summary: dict,
    ki_drift: dict,
    *,
    drift_tol: float,
) -> None:
    print("\n=== Scene-Shift Calibration Drift Baseline (Sub-step B) ===")
    print(f"Scene proxy: {split.get('scene_proxy', 'sensor_id')}")
    print(f"Strategy: {split.get('strategy')}")
    print(f"Calibration sensors: {split['calibration_sensors']}")
    print(f"Deployment sensors:  {split['deployment_sensors']}")
    print(f"\n{'Ki':<6} {'H_i':>6} {'P(IDK) cal':>12} {'P(IDK) dep':>12} {'delta':>8} "
          f"{'Acc cal':>10} {'Acc dep':>10}")
    print("-" * 72)
    for row in drift_rows:
        hi = row.get("H_i")
        hi_s = f"{hi:.2f}" if hi is not None else "—"
        p_cal = row.get("p_idk_calibration")
        p_dep = row.get("p_idk_deployment")
        delta = row.get("p_idk_delta")
        acc_cal = row.get("non_idk_acc_calibration")
        acc_dep = row.get("non_idk_acc_deployment")
        print(
            f"{row['ki']:<6} {hi_s:>6} "
            f"{p_cal if p_cal is not None else 0:12.4f} "
            f"{p_dep if p_dep is not None else 0:12.4f} "
            f"{delta if delta is not None else 0:8.4f} "
            f"{acc_cal if acc_cal is not None else 0:10.4f} "
            f"{acc_dep if acc_dep is not None else 0:10.4f}"
        )

    cal_c = cascade_summary["calibration"]
    dep_c = cascade_summary["deployment"]
    print("\nEXPAND cascade (fixed H_i, no recalibration):")
    print(f"  Calibration: n={cal_c['n']:,}  acc={cal_c['accuracy']:.4f}  "
          f"C_bar={cal_c['latency_ms']['mean']:.1f} ms  Kdet={cal_c['kdet_rate']:.4f}")
    print(f"  Deployment:  n={dep_c['n']:,}  acc={dep_c['accuracy']:.4f}  "
          f"C_bar={dep_c['latency_ms']['mean']:.1f} ms  Kdet={dep_c['kdet_rate']:.4f}")

    detected = drift_detected(ki_drift, tol=drift_tol)
    verdict = (
        "YES — per-Ki P(IDK) drift exceeds tolerance"
        if detected
        else "NO — P(IDK) stable across scenes (within tolerance)"
    )
    print(f"\nDrift detected (|delta P(IDK)| > {drift_tol}): {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile calibration drift under scene shift with fixed H_i",
    )
    parser.add_argument("--scene-inventory", type=Path, default=Path("checkpoints/scene_inventory.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/scene_shift_baseline.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("checkpoints/scene_shift_ki_drift.csv"))
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--calibration-sensors", type=str, default=None)
    parser.add_argument("--deploy-sensors", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample each scene group independently (smoke test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-tol", type=float, default=0.005)
    parser.add_argument("--timing", choices=("table", "live"), default="table")
    parser.add_argument("--kdet-mode", choices=KDET_MODES, default="registry")
    parser.add_argument("--plot", action="store_true", help="Write WIP PNG figures")
    args = parser.parse_args()

    cal_override = _parse_sensor_list(args.calibration_sensors)
    dep_override = _parse_sensor_list(args.deploy_sensors)
    if (cal_override is None) != (dep_override is None):
        parser.error("Provide both --calibration-sensors and --deploy-sensors, or neither.")

    split = load_scene_split(
        args.scene_inventory.resolve(),
        calibration_sensors=cal_override,
        deployment_sensors=dep_override,
    )

    bundle = load_eval_bundle(
        plan_path=args.plan.resolve(),
        processed_dir=args.processed_dir.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        seed=args.seed,
    )

    mic, geo, metadata = load_spectrogram_cache(args.processed_dir.resolve())
    cal_idx, dep_idx, cal_mask, dep_mask = build_scene_index_sets(
        metadata,
        split["calibration_sensors"],
        split["deployment_sensors"],
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(f"Hardware: {hardware_info()}")
    print(f"Calibration indices: {len(cal_idx):,}  Deployment indices: {len(dep_idx):,}")

    ki_drift = profile_all_kis(
        bundle.models,
        bundle.registry,
        mic,
        geo,
        metadata,
        cal_idx,
        dep_idx,
        bundle.device,
        batch_size=args.batch_size,
    )
    drift_rows = ki_drift_summary(ki_drift)

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
    cascade_summary = compare_cascade_scenes(bundle, cal_idx, dep_idx, config)

    detected = drift_detected(ki_drift, tol=args.drift_tol)
    payload = {
        "scene_split": split,
        "n_calibration": int(len(cal_idx)),
        "n_deployment": int(len(dep_idx)),
        "ki_drift": ki_drift,
        "ki_drift_summary": drift_rows,
        "cascade_summary": cascade_summary,
        "drift_detected": detected,
        "drift_tolerance": args.drift_tol,
        "hardware": hardware_info(),
        "timing_mode": args.timing,
        "kdet_mode": args.kdet_mode,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(drift_rows).to_csv(args.out_csv, index=False)

    fig_paths: dict[str, str] = {}
    if args.plot:
        fig_paths = write_all_figures(ki_drift, drift_rows, cascade_summary, args.fig_dir)
        payload["figure_paths"] = fig_paths
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_console_summary(split, drift_rows, cascade_summary, ki_drift, drift_tol=args.drift_tol)
    print(f"\nWrote {args.out_json.resolve()}")
    print(f"Wrote {args.out_csv.resolve()}")
    if fig_paths:
        print("Figures:")
        for name, path in fig_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
