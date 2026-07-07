"""Calibrate H_i on validation data for trained K0–K6 (no weight retraining).

Paper method: find lowest threshold such that non-IDK precision meets
required level (0.95 intermediate/specialized, 0.90 global).

Run:
  python calibrate_ki_thresholds.py
  python calibrate_ki_thresholds.py --ki K1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from models.dual_modal_cnn import build_ki_model
from training.calibration import calibrate_ki_threshold
from training.trainer import (
    build_loaders,
    get_device,
    load_spectrogram_cache,
    prepare_ki_arrays,
)
from utils.classifier_registry import ClassifierRegistry
from utils.labels import KI_REGISTRY, is_deterministic_ki, threshold_hi_for_ki

IDK_KIS = [f"K{i}" for i in range(7)]


def load_trained_model(ki_name: str, checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = checkpoint_dir / f"{ki_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")
    spec = KI_REGISTRY[ki_name]
    model = build_ki_model(ki_name, len(spec.class_names)).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def calibrate_one_ki(
    ki_name: str,
    *,
    processed_dir: Path,
    checkpoint_dir: Path,
    registry: ClassifierRegistry,
    cache,
    seed: int,
) -> dict:
    spec = KI_REGISTRY[ki_name]
    metrics_path = checkpoint_dir / f"{ki_name}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics {metrics_path}")

    prior = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss_key = prior.get("loss_key", "weighted_ce")
    device = get_device()

    arrays = prepare_ki_arrays(spec, processed_dir, cache=cache)
    _, val_loader, _, _, _, _ = build_loaders(
        spec, processed_dir, batch_size=128, loss_key=loss_key, arrays=arrays, seed=seed,
    )
    model = load_trained_model(ki_name, checkpoint_dir, device)
    required_precision = float(threshold_hi_for_ki(ki_name))
    hi, p_idk, cal_meta = calibrate_ki_threshold(
        model, val_loader, device, spec.modality, required_precision,
    )

    prior["threshold_hi"] = hi
    prior["p_idk"] = p_idk
    prior["calibration"] = cal_meta
    metrics_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")

    rec = registry.get(ki_name)
    if rec is not None:
        registry.upsert(replace(rec, threshold_hi=hi, p_idk=p_idk))

    print(
        f"{ki_name}: H_i {hi:.4f}  required={required_precision:.2f}  "
        f"achieved={cal_meta['achieved_precision']:.4f}  p_idk={p_idk:.4f}"
    )
    return {"ki": ki_name, "threshold_hi": hi, "p_idk": p_idk, **cal_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate IDK thresholds on validation data")
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--ki", default="all", help="Ki name or 'all' for K0–K6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    targets = IDK_KIS if args.ki == "all" else [args.ki]
    for ki in targets:
        if is_deterministic_ki(ki):
            parser.error(f"{ki} is deterministic — no H_i calibration")

    registry = (
        ClassifierRegistry.load(args.registry)
        if args.registry.exists()
        else ClassifierRegistry.from_checkpoint_dir(args.checkpoint_dir)
    )
    cache = load_spectrogram_cache(args.processed_dir)

    rows = []
    for ki in targets:
        rows.append(
            calibrate_one_ki(
                ki,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                registry=registry,
                cache=cache,
                seed=args.seed,
            )
        )

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    registry.save(args.registry)

    out = args.checkpoint_dir / "threshold_calibration.json"
    out.write_text(json.dumps({"classifiers": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.registry.resolve()}")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
