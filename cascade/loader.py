"""Load trained Ki classifiers from checkpoint or flat weight files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.dual_modal_cnn import build_ki_model
from training.trainer import get_device
from utils.classifier_registry import ClassifierRegistry
from utils.labels import KI_REGISTRY


def _resolve_path(path_str: str, checkpoint_dir: Path) -> Path:
    """Normalize registry paths (Windows backslashes) and resolve relative paths."""
    p = Path(path_str.replace("\\", "/"))
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "checkpoints":
        return Path(*p.parts)
    return checkpoint_dir / p.name


def _weights_only_path(checkpoint_dir: Path, ki_name: str) -> Path:
    return checkpoint_dir / "weights" / f"{ki_name}_weights.pt"


def load_state_dict_for_ki(
    ki_name: str,
    checkpoint_path: Path,
    weights_path: Path | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a state_dict from flat weights file or full training checkpoint.

    Full checkpoints store tensors under ``model_state_dict``.
    Flat ``*_weights.pt`` files store the state_dict at the top level.
    """
    dev = device or torch.device("cpu")

    if weights_path is not None and weights_path.exists():
        state_dict = torch.load(weights_path, map_location=dev, weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            return state_dict["model_state_dict"]
        return state_dict

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No weights for {ki_name}: missing {checkpoint_path} "
            f"and {weights_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise ValueError(
        f"{checkpoint_path} does not contain model weights. "
        f"Expected key 'model_state_dict' or a flat state_dict. "
        f"Found keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}. "
        f"Run: python export_model_weights.py"
    )


def load_cascade_models(
    checkpoint_dir: Path,
    registry_path: Path,
    *,
    prefer_flat_weights: bool = True,
) -> tuple[dict[str, nn.Module], ClassifierRegistry, torch.device]:
    """Load all Ki models listed in KI_REGISTRY using registry metadata."""
    registry = ClassifierRegistry.load(registry_path)
    device = get_device()
    models: dict[str, nn.Module] = {}

    for ki_name in KI_REGISTRY:
        rec = registry.get(ki_name)
        if rec is None or not rec.checkpoint:
            raise ValueError(f"No registry record for {ki_name}")

        ckpt_path = _resolve_path(rec.checkpoint, checkpoint_dir)
        weights_path = _weights_only_path(checkpoint_dir, ki_name)

        if prefer_flat_weights and not weights_path.exists():
            weights_path = None
        elif not prefer_flat_weights:
            weights_path = None

        state_dict = load_state_dict_for_ki(
            ki_name,
            ckpt_path,
            weights_path=weights_path,
            device=device,
        )

        model = build_ki_model(ki_name, len(rec.class_names)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models[ki_name] = model

    return models, registry, device
