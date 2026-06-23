"""Export flat state_dict weight files from full Ki checkpoints.

Full training checkpoints (checkpoints/K0.pt) wrap weights inside
``model_state_dict``. Teammates often call ``load_state_dict(torch.load(...))``
directly on the file, which fails because the outer object is a dict of metadata.

This script writes:
  checkpoints/weights/K0_weights.pt  …  one state_dict per Ki (tensors only)
  checkpoints/weights/all_weights.pt   …  {Ki_name: state_dict} bundle
  checkpoints/weights/manifest.json    …  paths, class names, param counts, SHA256
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from utils.labels import KI_REGISTRY

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_full_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected dict checkpoint, got {type(ckpt)}")
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"{path}: missing 'model_state_dict'. Keys found: {list(ckpt.keys())}"
        )
    return ckpt


def export_weights(checkpoint_dir: Path, weights_dir: Path) -> dict:
    weights_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, dict] = {}
    manifest_rows: list[dict] = []

    for ki_name in KI_NAMES:
        ckpt_path = checkpoint_dir / f"{ki_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = _load_full_checkpoint(ckpt_path)
        state_dict = ckpt["model_state_dict"]
        spec = KI_REGISTRY[ki_name]

        weights_path = weights_dir / f"{ki_name}_weights.pt"
        torch.save(state_dict, weights_path)

        num_params = sum(t.numel() for t in state_dict.values())
        bundle[ki_name] = state_dict

        row = {
            "name": ki_name,
            "level": spec.level,
            "modality": ckpt.get("modality", spec.modality),
            "class_names": ckpt.get("class_names", spec.class_names),
            "num_parameters": num_params,
            "num_tensors": len(state_dict),
            "loss_key": ckpt.get("loss_key"),
            "val_macro_f1": ckpt.get("val_macro_f1"),
            "full_checkpoint": f"checkpoints/{ki_name}.pt",
            "weights_file": f"checkpoints/weights/{ki_name}_weights.pt",
            "sha256_weights": _sha256(weights_path),
            "sha256_full": _sha256(ckpt_path),
        }
        manifest_rows.append(row)
        print(f"{ki_name}: exported {num_params:,} params -> {weights_path.name}")

    bundle_path = weights_dir / "all_weights.pt"
    torch.save(bundle, bundle_path)

    manifest = {
        "format_version": 1,
        "description": (
            "Flat PyTorch state_dict files for K0-K6 and Kdet. "
            "Load with model.load_state_dict(torch.load(path, map_location=device))."
        ),
        "bundle_file": "checkpoints/weights/all_weights.pt",
        "load_example": (
            "weights = torch.load('checkpoints/weights/K0_weights.pt', map_location='cpu'); "
            "model.load_state_dict(weights)"
        ),
        "classifiers": manifest_rows,
    }
    manifest_path = weights_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote bundle {bundle_path} ({len(bundle)} classifiers)")
    print(f"Wrote manifest {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export flat state_dict weight files")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--weights-dir", type=Path, default=Path("checkpoints/weights"))
    args = parser.parse_args()
    export_weights(args.checkpoint_dir, args.weights_dir)


if __name__ == "__main__":
    main()
