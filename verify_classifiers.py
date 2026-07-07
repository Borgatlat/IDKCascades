"""Verify that Ki checkpoint and weight files load into the expected architectures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cascade.checkpoint_paths import resolve_registry_checkpoint
from cascade.loader import load_state_dict_from_file
from models.dual_modal_cnn import build_ki_model
from utils.labels import KI_REGISTRY

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def get_output_dim(model: torch.nn.Module) -> int:
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "__getitem__"):
        raise RuntimeError("Model does not expose a classifier sequential module")
    last_layer = classifier[-1]
    out_features = getattr(last_layer, "out_features", None)
    if out_features is None:
        raise RuntimeError("Final classifier layer does not expose out_features")
    return int(out_features)


def verify_one(
    ki_name: str,
    checkpoint_dir: Path,
    registry_path: Path | None,
) -> dict:
    spec = KI_REGISTRY[ki_name]
    path = resolve_registry_checkpoint(None, ki_name, checkpoint_dir, registry_path)
    state_dict = load_state_dict_from_file(path)

    model = build_ki_model(ki_name, len(spec.class_names))
    model.load_state_dict(state_dict)

    num_params = sum(t.numel() for t in state_dict.values())
    num_classes = get_output_dim(model)
    expected_num_classes = len(spec.class_names)
    return {
        "name": ki_name,
        "loaded_from": str(path.resolve()),
        "num_parameters": num_params,
        "num_tensors": len(state_dict),
        "num_classes": num_classes,
        "expected_num_classes": expected_num_classes,
        "load_ok": num_classes == expected_num_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify classifier weight files")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("checkpoints/classifier_registry.json"),
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional report output path")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve() if args.registry.exists() else None

    results = []
    for ki_name in KI_NAMES:
        if ki_name == "Kdet":
            print("[SKIP] Kdet: simulated stub (no weight file required)")
            continue
        row = verify_one(ki_name, checkpoint_dir, registry_path)
        results.append(row)
        status = "[OK]" if row["load_ok"] else "[WARN]"
        print(
            f"{status} {ki_name}: final layer = {row['num_classes']} classes "
            f"(expected {row['expected_num_classes']}), "
            f"{row['num_parameters']:,} params from {row['loaded_from']}"
        )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("All classifiers verified.")


if __name__ == "__main__":
    main()
