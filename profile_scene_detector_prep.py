"""Day 1–2 scene detector prep (Person A): audit, holdout, profiles, baseline B.

Does not run process_scenes.py — assumes preprocessing is handled separately.

Examples:
  python profile_scene_detector_prep.py --audit
  python profile_scene_detector_prep.py --holdout --deployment-scene i22
  python profile_scene_detector_prep.py --profiles --max-samples 500
  python profile_scene_detector_prep.py --separability
  python profile_scene_detector_prep.py --baseline-b --deployment-scene i22
  python profile_scene_detector_prep.py --day1-day2 --max-samples 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cascade.eval import evaluate_val_set, hardware_info, summarize_traces
from cascade.executor import CascadePlan, ExecutorConfig, load_wcet_profile
from cascade.kdet import KDET_MODES, build_kdet_context
from cascade.loader import load_cascade_models
from process_data import KNOWN_SCENES
from utils.scene_data import (
    DEFAULT_HOLDOUT_SCENE,
    audit_scene_readiness,
    build_stratified_holdout_indices,
    eval_mask_for_scene,
    load_holdout_eval_indices,
    load_scene_cache,
    save_holdout_artifacts,
)
from utils.scene_profiles import (
    build_all_scene_profiles,
    profile_separability_matrix,
    write_profiles_json,
)


def _filter_indices(metadata, indices, scene_id: str):
    mask = eval_mask_for_scene(metadata, scene_id)
    return indices[mask[indices]]


def cmd_audit(processed_root: Path, out_json: Path) -> dict:
    payload = audit_scene_readiness(processed_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Data readiness: {payload['n_ready']}/{payload['n_total']} scenes processed")
    for row in payload["scenes"]:
        status = "OK" if row["ready"] else "MISSING"
        print(f"  [{status}] {row['scene_id']}: {row['n_segments']:,} segments")
    print(f"Wrote {out_json.resolve()}")
    return payload


def cmd_holdout(
    deployment_scene: str,
    processed_root: Path,
    checkpoints_dir: Path,
    seed: int,
) -> None:
    _mic, _geo, metadata = load_scene_cache(deployment_scene, processed_root)
    eval_idx, reserve_idx = build_stratified_holdout_indices(
        metadata, deployment_scene, seed=seed,
    )
    paths = save_holdout_artifacts(
        metadata, deployment_scene, eval_idx, reserve_idx, checkpoints_dir,
    )
    print(
        f"Holdout {deployment_scene}: eval={len(eval_idx):,} reserve={len(reserve_idx):,}"
    )
    for name, path in paths.items():
        print(f"  {name}: {path.resolve()}")


def cmd_profiles(
    *,
    processed_root: Path,
    checkpoint_dir: Path,
    registry_path: Path,
    out_json: Path,
    out_csv: Path,
    scene_ids: list[str],
    batch_size: int,
    max_samples: int | None,
    seed: int,
) -> dict:
    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)
    payload = build_all_scene_profiles(
        models,
        registry,
        scene_ids,
        device,
        processed_root=processed_root,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
    )
    write_profiles_json(payload, out_json)
    print(f"Built profiles for {len(payload['profiles'])} scenes: {sorted(payload['profiles'])}")
    if payload["skipped_scenes"]:
        print("Skipped (not processed yet):")
        for row in payload["skipped_scenes"]:
            print(f"  - {row['scene_id']}")

    sep = profile_separability_matrix(payload)
    sep.to_csv(out_csv)
    print(f"Wrote {out_json.resolve()}")
    print(f"Wrote {out_csv.resolve()}")
    return payload


def cmd_baseline_b(
    *,
    deployment_scene: str,
    processed_root: Path,
    plan_path: Path,
    checkpoint_dir: Path,
    registry_path: Path,
    checkpoints_dir: Path,
    timing: str,
    kdet_mode: str,
    batch_size: int,
    max_samples: int | None,
    seed: int,
) -> dict:
    from utils.scene_data import FORBIDDEN_GLOBAL_LABELS

    mic, geo, metadata = load_scene_cache(deployment_scene, processed_root)
    holdout = load_holdout_eval_indices(deployment_scene, checkpoints_dir)
    holdout = _filter_indices(metadata, holdout, deployment_scene)
    if max_samples is not None:
        holdout = holdout[:max_samples]

    plan = CascadePlan.load(plan_path)
    wcet_ms = load_wcet_profile(checkpoint_dir)
    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)

    kdet = build_kdet_context(
        kdet_mode,
        metrics_path=checkpoint_dir / "Kdet_metrics.json",
        checkpoint_dir=checkpoint_dir,
        registry_path=registry_path,
        wcet_ms=wcet_ms.get("Kdet"),
    )
    config = ExecutorConfig(
        timing_mode=timing,
        wcet_ms=wcet_ms,
        deadline_ms=None,
        deadline_guard=False,
        kdet=kdet,
    )
    traces = evaluate_val_set(
        plan,
        models,
        registry,
        mic,
        geo,
        metadata,
        holdout,
        device,
        config,
    )
    summary = summarize_traces(traces)
    payload = {
        "condition": "B_fixed_thresholds",
        "deployment_scene": deployment_scene,
        "n_holdout": int(len(holdout)),
        "forbidden_labels": sorted(FORBIDDEN_GLOBAL_LABELS.get(deployment_scene, frozenset())),
        "cascade_summary": summary,
        "hardware": hardware_info(),
        "timing_mode": timing,
        "kdet_mode": kdet_mode,
    }
    out_json = checkpoints_dir / "scene_detector_baseline_fixed.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Condition B ({deployment_scene}): n={summary['n']:,} "
        f"acc={summary['accuracy']:.4f} kdet={summary['kdet_rate']:.4f} "
        f"C_bar={summary['latency_ms']['mean']:.1f} ms"
    )
    print(f"Wrote {out_json.resolve()}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene detector Day 1–2 prep (Person A)")
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--plan", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--deployment-scene", type=str, default=DEFAULT_HOLDOUT_SCENE)
    parser.add_argument("--scenes", type=str, default=",".join(KNOWN_SCENES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timing", choices=("table", "live"), default="table")
    parser.add_argument("--kdet-mode", choices=KDET_MODES, default="registry")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--separability", action="store_true", help="Requires profiles JSON")
    parser.add_argument("--baseline-b", action="store_true")
    parser.add_argument("--day1-day2", action="store_true", help="audit + holdout + profiles + separability + baseline B")
    parser.add_argument(
        "--profiles-json",
        type=Path,
        default=Path("checkpoints/scene_confidence_profiles.json"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("checkpoints/scene_data_readiness.json"),
    )
    parser.add_argument(
        "--separability-csv",
        type=Path,
        default=Path("checkpoints/scene_profile_separability.csv"),
    )
    args = parser.parse_args()

    scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    run_all = args.day1_day2
    if run_all:
        args.audit = True
        args.holdout = True
        args.profiles = True
        args.baseline_b = True

    if args.audit:
        cmd_audit(args.processed_root, args.audit_json)

    if args.holdout:
        try:
            cmd_holdout(
                args.deployment_scene, args.processed_root, args.checkpoint_dir, args.seed,
            )
        except FileNotFoundError as exc:
            print(f"[holdout] SKIP: {exc}")
            print("Re-run when deployment scene preprocessing finishes.")

    if args.profiles:
        cmd_profiles(
            processed_root=args.processed_root,
            checkpoint_dir=args.checkpoint_dir,
            registry_path=args.registry,
            out_json=args.profiles_json,
            out_csv=args.separability_csv,
            scene_ids=scene_ids,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    elif args.separability:
        payload = json.loads(args.profiles_json.read_text(encoding="utf-8"))
        sep = profile_separability_matrix(payload)
        sep.to_csv(args.separability_csv)
        print(f"Wrote {args.separability_csv.resolve()}")

    if args.baseline_b:
        try:
            cmd_baseline_b(
                deployment_scene=args.deployment_scene,
                processed_root=args.processed_root,
                plan_path=args.plan,
                checkpoint_dir=args.checkpoint_dir,
                registry_path=args.registry,
                checkpoints_dir=args.checkpoint_dir,
                timing=args.timing,
                kdet_mode=args.kdet_mode,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        except FileNotFoundError as exc:
            print(f"[baseline-b] SKIP: {exc}")


if __name__ == "__main__":
    main()
