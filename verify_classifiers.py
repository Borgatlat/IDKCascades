"""Verify that Ki checkpoint and weight files load into the expected architectures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.dual_modal_cnn import build_ki_model
from utils.labels import KI_REGISTRY

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def _load_state_dict_from_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"{path}: unrecognized checkpoint format")


def verify_one(ki_name: str, checkpoint_dir: Path, weights_dir: Path) -> dict:
    spec = KI_REGISTRY[ki_name]
    model = build_ki_model(ki_name, len(spec.class_names))

    full_path = checkpoint_dir / f"{ki_name}.pt"
    weights_path = weights_dir / f"{ki_name}_weights.pt"

    full_sd = _load_state_dict_from_checkpoint(full_path)
    flat_sd = _load_state_dict_from_checkpoint(weights_path)

    model.load_state_dict(full_sd)
    model2 = build_ki_model(ki_name, len(spec.class_names))
    model2.load_state_dict(flat_sd)

    full_params = sum(t.numel() for t in full_sd.values())
    flat_params = sum(t.numel() for t in flat_sd.values())
    keys_match = list(full_sd.keys()) == list(flat_sd.keys())
    values_match = keys_match and all(
        torch.equal(full_sd[k], flat_sd[k]) for k in full_sd.keys()
    )

    return {
        "name": ki_name,
        "full_checkpoint": str(full_path),
        "weights_file": str(weights_path),
        "num_parameters": full_params,
        "flat_matches_full": values_match,
        "load_ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify classifier weight files")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--weights-dir", type=Path, default=Path("checkpoints/weights"))
    parser.add_argument("--json", type=Path, default=None, help="Optional report output path")
    args = parser.parse_args()

    results = []
    ok = True
    for ki_name in KI_NAMES:
        row = verify_one(ki_name, args.checkpoint_dir, args.weights_dir)
        results.append(row)
        status = "OK" if row["flat_matches_full"] else "MISMATCH"
        if not row["flat_matches_full"]:
            ok = False
        print(
            f"[{status}] {ki_name}: {row['num_parameters']:,} params, "
            f"flat weights match full checkpoint={row['flat_matches_full']}"
        )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not ok:
        raise SystemExit(1)
    print("All classifiers verified.")


if __name__ == "__main__":
    main()
