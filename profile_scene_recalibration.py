"""Sub-step C: deployment-scene threshold re-tuning (match calibration P(IDK)).

Run:
  python profile_scene_recalibration.py --plot
  python profile_scene_recalibration.py --max-samples 500 --plot
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from cascade.eval import hardware_info, load_eval_bundle
from cascade.executor import ExecutorConfig
from cascade.kdet import KDET_MODES, build_kdet_context
from utils.scene_calibration import (
    build_scene_index_sets,
    compare_cascade_on_indices,
    load_baseline_targets,
    load_scene_split,
    profile_kis_on_indices,
    recalibration_ki_summary,
    registry_with_thresholds,
    split_tune_eval_indices,
    tune_thresholds_all_kis,
    write_recalibration_figures,
)

DEFAULT_FIG_DIR = Path("checkpoints/figures/scene_recalibration")


def _parse_sensor_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def print_console_summary(
    split: dict,
    tune_rows: list[dict],
    ki_rows: list[dict],
    cascade_fixed: dict,
    cascade_tuned: dict,
    *,
    n_tune: int,
    n_eval: int,
) -> None:
    print("\n=== Deployment-Scene Threshold Re-Tuning (Sub-step C) ===")
    print(f"Scene proxy: {split.get('scene_proxy', 'sensor_id')}")
    print(f"Deployment sensors: {split['deployment_sensors']}")
    print(f"Tune / eval split: {n_tune:,} / {n_eval:,} deployment segments")

    print(f"\n{'Ki':<6} {'H fixed':>8} {'H tuned':>8} {'target':>8} "
          f"{'P(IDK) fix':>11} {'P(IDK) new':>11}")
    print("-" * 68)
    for row in ki_rows:
        print(
            f"{row['ki']:<6} "
            f"{row.get('H_i_fixed', 0):8.4f} "
            f"{row.get('H_i_tuned', 0):8.4f} "
            f"{row.get('target_p_idk', 0):8.4f} "
            f"{row.get('p_idk_fixed_eval', 0):11.4f} "
            f"{row.get('p_idk_tuned_eval', 0):11.4f}"
        )

    print("\nEXPAND cascade on deployment eval holdout:")
    print(f"  Fixed H_i:   n={cascade_fixed['n']:,}  acc={cascade_fixed['accuracy']:.4f}  "
          f"C_bar={cascade_fixed['latency_ms']['mean']:.1f} ms  "
          f"Kdet={cascade_fixed['kdet_rate']:.4f}")
    print(f"  Retuned H_i: n={cascade_tuned['n']:,}  acc={cascade_tuned['accuracy']:.4f}  "
          f"C_bar={cascade_tuned['latency_ms']['mean']:.1f} ms  "
          f"Kdet={cascade_tuned['kdet_rate']:.4f}")

    acc_delta = cascade_tuned["accuracy"] - cascade_fixed["accuracy"]
    kdet_delta = cascade_tuned["kdet_rate"] - cascade_fixed["kdet_rate"]
    lat_delta = cascade_tuned["latency_ms"]["mean"] - cascade_fixed["latency_ms"]["mean"]
    print(f"\nDelta (retuned - fixed): acc={acc_delta:+.4f}  "
          f"Kdet={kdet_delta:+.4f}  C_bar={lat_delta:+.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-tune H_i on deployment scene to match calibration P(IDK)",
    )
    parser.add_argument("--scene-inventory", type=Path, default=Path("checkpoints/scene_inventory.json"))
    parser.add_argument("--baseline-json", type=Path, default=Path("checkpoints/scene_shift_baseline.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/scene_recalibration.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("checkpoints/scene_recalibration_ki.csv"))
    parser.add_argument("--overrides-json", type=Path, default=Path("checkpoints/scene_threshold_overrides.json"))
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--calibration-sensors", type=str, default=None)
    parser.add_argument("--deploy-sensors", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample deployment pool before tune/eval split (smoke test)")
    parser.add_argument("--tune-fraction", type=float, default=0.20)
    parser.add_argument("--tune-samples", type=int, default=None,
                        help="Cap tune-set size (eval gets the rest)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
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
    target_p_idk = load_baseline_targets(args.baseline_json.resolve())

    bundle = load_eval_bundle(
        plan_path=args.plan.resolve(),
        processed_dir=args.processed_dir.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        registry_path=args.registry.resolve(),
        seed=args.seed,
    )

    mic, geo, metadata = bundle.mic, bundle.geo, bundle.metadata
    _, dep_idx, _, _ = build_scene_index_sets(
        metadata,
        split["calibration_sensors"],
        split["deployment_sensors"],
        max_samples=args.max_samples,
        seed=args.seed,
    )
    tune_idx, eval_idx = split_tune_eval_indices(
        dep_idx,
        tune_fraction=args.tune_fraction,
        tune_samples=args.tune_samples,
        seed=args.seed,
    )

    print(f"Hardware: {hardware_info()}")
    print(f"Deployment pool: {len(dep_idx):,}  tune: {len(tune_idx):,}  eval: {len(eval_idx):,}")

    overrides, tune_rows = tune_thresholds_all_kis(
        bundle.models,
        bundle.registry,
        mic,
        geo,
        metadata,
        tune_idx,
        target_p_idk,
        bundle.device,
        batch_size=args.batch_size,
    )

    tuned_registry = registry_with_thresholds(bundle.registry, overrides)

    fixed_profiles = profile_kis_on_indices(
        bundle.models, bundle.registry, mic, geo, metadata, eval_idx, bundle.device,
        batch_size=args.batch_size,
    )
    tuned_profiles = profile_kis_on_indices(
        bundle.models, tuned_registry, mic, geo, metadata, eval_idx, bundle.device,
        batch_size=args.batch_size,
    )
    ki_rows = recalibration_ki_summary(fixed_profiles, tuned_profiles, tune_rows, target_p_idk)

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
    cascade_fixed = compare_cascade_on_indices(bundle, eval_idx, config)
    cascade_tuned = compare_cascade_on_indices(bundle, eval_idx, config, registry=tuned_registry)

    payload = {
        "scene_split": split,
        "baseline_json": str(args.baseline_json.resolve()),
        "target_p_idk": target_p_idk,
        "threshold_overrides": overrides,
        "tune_fraction": args.tune_fraction,
        "n_deployment_pool": int(len(dep_idx)),
        "n_tune": int(len(tune_idx)),
        "n_eval": int(len(eval_idx)),
        "tune_rows": tune_rows,
        "ki_summary": ki_rows,
        "cascade_eval_fixed": cascade_fixed,
        "cascade_eval_tuned": cascade_tuned,
        "hardware": hardware_info(),
        "timing_mode": args.timing,
        "kdet_mode": args.kdet_mode,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(ki_rows).to_csv(args.out_csv, index=False)
    args.overrides_json.write_text(
        json.dumps({"threshold_overrides": overrides, "target_p_idk": target_p_idk}, indent=2),
        encoding="utf-8",
    )

    fig_paths: dict[str, str] = {}
    if args.plot:
        gc.collect()
        fig_paths = write_recalibration_figures(
            tune_rows, ki_rows, cascade_fixed, cascade_tuned, args.fig_dir,
        )
        payload["figure_paths"] = fig_paths
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_console_summary(
        split, tune_rows, ki_rows, cascade_fixed, cascade_tuned,
        n_tune=len(tune_idx), n_eval=len(eval_idx),
    )
    print(f"\nWrote {args.out_json.resolve()}")
    print(f"Wrote {args.out_csv.resolve()}")
    print(f"Wrote {args.overrides_json.resolve()}")
    if fig_paths:
        print("Figures:")
        for name, path in fig_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
