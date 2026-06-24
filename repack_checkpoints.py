"""Repack K0-K6 and Kdet checkpoints so weights are guaranteed and paths are standardized."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from cascade.checkpoint_paths import relative_checkpoint_ref
from utils.labels import KI_REGISTRY

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def _extract_state_dict(ckpt: object) -> dict:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


def repack_one(
    ki_name: str,
    checkpoint_dir: Path,
    *,
    write_pth_copy: bool = True,
) -> dict:
    """Rewrite Ki.pt with explicit model_state_dict + metadata; mirror .pth copy."""
    src = checkpoint_dir / f"{ki_name}.pt"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    raw = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"{src}: expected dict checkpoint")

    state_dict = _extract_state_dict(raw)
    spec = KI_REGISTRY[ki_name]

    payload = {
        "ki": ki_name,
        "loss_key": raw.get("loss_key"),
        "class_names": raw.get("class_names", spec.class_names),
        "modality": raw.get("modality", spec.modality),
        "num_classes": len(spec.class_names),
        "val_macro_f1": raw.get("val_macro_f1"),
        "val_macro_f1_full": raw.get("val_macro_f1_full"),
        "model_state_dict": state_dict,
    }

    torch.save(payload, src)
    if write_pth_copy:
        shutil.copy2(src, checkpoint_dir / f"{ki_name}.pth")

    weights_dir = checkpoint_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, weights_dir / f"{ki_name}_weights.pt")
    if write_pth_copy:
        shutil.copy2(weights_dir / f"{ki_name}_weights.pt", weights_dir / f"{ki_name}_weights.pth")

    num_params = sum(t.numel() for t in state_dict.values())
    return {
        "name": ki_name,
        "checkpoint": relative_checkpoint_ref(ki_name),
        "num_parameters": num_params,
        "num_tensors": len(state_dict),
        "pt_file": str(src.resolve()),
    }


def fix_registry_paths(checkpoint_dir: Path) -> None:
    """Set registry checkpoint fields to simple Ki.pt filenames."""
    registry_path = checkpoint_dir / "classifier_registry.json"
    if not registry_path.exists():
        return

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    for row in data.get("classifiers", []):
        name = row.get("name")
        if name in KI_NAMES:
            row["checkpoint"] = relative_checkpoint_ref(name)
    registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated registry paths -> {registry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack Ki checkpoints with guaranteed weights")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--no-pth", action="store_true", help="Skip .pth mirror copies")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    report = []
    for ki_name in KI_NAMES:
        row = repack_one(ki_name, checkpoint_dir, write_pth_copy=not args.no_pth)
        report.append(row)
        print(f"{ki_name}: repacked {row['num_parameters']:,} params -> {row['checkpoint']}")

    fix_registry_paths(checkpoint_dir)

    bundle: dict[str, dict] = {}
    for ki_name in KI_NAMES:
        bundle[ki_name] = torch.load(
            checkpoint_dir / "weights" / f"{ki_name}_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
    torch.save(bundle, checkpoint_dir / "weights" / "all_weights.pt")

    report_path = checkpoint_dir / "weights" / "repack_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
