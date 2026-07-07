"""14-dim confidence / accept-rate profiles per M3N-VC scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.scene_calibration import IDK_KI_NAMES, profile_kis_on_indices
from utils.scene_data import all_scene_indices, load_scene_cache

FEATURE_NAMES: list[str] = []
for _ki in IDK_KI_NAMES:
    FEATURE_NAMES.append(f"{_ki}_mean_confidence")
    FEATURE_NAMES.append(f"{_ki}_accept_rate")


def ki_profiles_to_vector(ki_profiles: dict[str, dict[str, Any]]) -> np.ndarray:
    """Flatten per-Ki stats into 14-dim vector (mean conf, accept rate)."""
    values: list[float] = []
    for ki_name in IDK_KI_NAMES:
        prof = ki_profiles.get(ki_name, {})
        mean_conf = float(prof.get("mean_confidence", 0.0) or 0.0)
        p_idk = prof.get("p_idk")
        accept = float(1.0 - p_idk) if p_idk is not None else 0.0
        values.extend([mean_conf, accept])
    return np.array(values, dtype=np.float64)


def build_scene_profile(
    models: dict[str, nn.Module | None],
    registry,
    scene_id: str,
    device: torch.device,
    *,
    processed_root: Path | str = "datasets/processed",
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute reference profile for one scene (all sensors pooled)."""
    mic, geo, metadata = load_scene_cache(scene_id, processed_root)
    indices = all_scene_indices(metadata)
    if max_samples is not None and len(indices) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))

    ki_profiles = profile_kis_on_indices(
        models, registry, mic, geo, metadata, indices, device, batch_size=batch_size,
    )
    vector = ki_profiles_to_vector(ki_profiles)
    return {
        "scene_id": scene_id,
        "vector": vector.tolist(),
        "feature_names": FEATURE_NAMES,
        "n_segments": int(len(indices)),
        "per_ki": ki_profiles,
    }


def build_all_scene_profiles(
    models: dict[str, nn.Module | None],
    registry,
    scene_ids: list[str],
    device: torch.device,
    *,
    processed_root: Path | str = "datasets/processed",
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Build profiles for every scene with processed data available."""
    profiles: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for scene_id in scene_ids:
        try:
            profiles[scene_id] = build_scene_profile(
                models,
                registry,
                scene_id,
                device,
                processed_root=processed_root,
                batch_size=batch_size,
                max_samples=max_samples,
                seed=seed,
            )
        except FileNotFoundError as exc:
            skipped.append({"scene_id": scene_id, "reason": str(exc)})

    matrix = np.stack(
        [np.array(profiles[s]["vector"], dtype=np.float64) for s in sorted(profiles)],
        axis=0,
    )
    z_mean = matrix.mean(axis=0) if len(matrix) else np.zeros(14)
    z_std = matrix.std(axis=0) if len(matrix) else np.ones(14)
    z_std = np.where(z_std < 1e-9, 1.0, z_std)

    return {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "normalization": {"z_mean": z_mean.tolist(), "z_std": z_std.tolist()},
        "profiles": profiles,
        "skipped_scenes": skipped,
    }


def profile_separability_matrix(payload: dict[str, Any]) -> pd.DataFrame:
    """Pairwise Euclidean distance between scene profile vectors."""
    scene_ids = sorted(payload["profiles"].keys())
    vectors = np.array(
        [payload["profiles"][s]["vector"] for s in scene_ids], dtype=np.float64,
    )
    z_mean = np.array(payload["normalization"]["z_mean"], dtype=np.float64)
    z_std = np.array(payload["normalization"]["z_std"], dtype=np.float64)
    normed = (vectors - z_mean) / z_std

    n = len(scene_ids)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.linalg.norm(normed[i] - normed[j]))

    return pd.DataFrame(dist, index=scene_ids, columns=scene_ids)


def write_profiles_json(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
