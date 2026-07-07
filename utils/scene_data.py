"""M3N-VC multi-scene data loading, eval masks, and holdout splits.

Scenes are deployment environments (h24, h08, …), not sensor nodes (rs1, rs2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_data import KNOWN_SCENES
from training.trainer import load_spectrogram_cache, normalize_spectrograms

# Per-scene labels that must be excluded from scoring (team spec).
FORBIDDEN_GLOBAL_LABELS: dict[str, frozenset[str]] = {
    "a06": frozenset({"mustang"}),
    "i22": frozenset({"gle350"}),
}

DEFAULT_HOLDOUT_SCENE = "i22"
CALIBRATION_SCENES = ("h08", "h24", "s31", "a06", "i29")
HOLDOUT_EVAL_FRACTION = 0.80


def scene_processed_dir(processed_root: Path, scene_id: str) -> Path:
    """Return folder containing spectrograms for one M3N-VC scene."""
    processed_root = Path(processed_root)
    if scene_id == "h24" and (processed_root / "h24_metadata.parquet").is_file():
        return processed_root
    return processed_root / scene_id


def scene_cache_paths(scene_dir: Path, scene_id: str) -> tuple[Path, Path, Path]:
    """Mic, geo, metadata paths for a scene (norm cache preferred)."""
    scene_dir = Path(scene_dir)
    prefix = scene_id
    norm_mic = scene_dir / f"{prefix}_paired_mic_norm.npy"
    norm_geo = scene_dir / f"{prefix}_paired_geo_norm.npy"
    meta = scene_dir / f"{prefix}_metadata.parquet"
    raw_mic = scene_dir / f"{prefix}_paired_mic.npy"
    raw_geo = scene_dir / f"{prefix}_paired_geo.npy"
    if norm_mic.is_file() and norm_geo.is_file() and meta.is_file():
        return norm_mic, norm_geo, meta
    if raw_mic.is_file() and raw_geo.is_file() and meta.is_file():
        return raw_mic, raw_geo, meta
    raise FileNotFoundError(
        f"Missing processed cache for {scene_id} under {scene_dir} "
        f"(expected {prefix}_metadata.parquet and paired mic/geo)."
    )


def load_scene_cache(
    scene_id: str,
    processed_root: Path | str = "datasets/processed",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load mic/geo spectrograms + metadata for one M3N-VC scene."""
    scene_dir = scene_processed_dir(Path(processed_root), scene_id)
    mic_path, geo_path, meta_path = scene_cache_paths(scene_dir, scene_id)

    if mic_path.name.endswith("_norm.npy"):
        mic = np.load(mic_path, mmap_mode="r")
        geo = np.load(geo_path, mmap_mode="r")
    else:
        mic = normalize_spectrograms(np.load(mic_path))
        geo = normalize_spectrograms(np.load(geo_path))
        norm_mic = scene_dir / f"{scene_id}_paired_mic_norm.npy"
        norm_geo = scene_dir / f"{scene_id}_paired_geo_norm.npy"
        np.save(norm_mic, mic)
        np.save(norm_geo, geo)

    metadata = pd.read_parquet(meta_path)
    if "scene_id" not in metadata.columns:
        metadata = metadata.copy()
        metadata["scene_id"] = scene_id
    return mic, geo, metadata


def audit_scene_readiness(
    processed_root: Path | str = "datasets/processed",
    scene_ids: tuple[str, ...] = KNOWN_SCENES,
) -> dict[str, Any]:
    """Check which scenes have processed spectrogram caches (no preprocessing)."""
    processed_root = Path(processed_root)
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_dir = scene_processed_dir(processed_root, scene_id)
        try:
            _, _, meta_path = scene_cache_paths(scene_dir, scene_id)
            n_segments = len(pd.read_parquet(meta_path))
            ready = True
            err = None
        except FileNotFoundError as exc:
            n_segments = 0
            ready = False
            err = str(exc)
        rows.append(
            {
                "scene_id": scene_id,
                "ready": ready,
                "processed_dir": str(scene_dir),
                "n_segments": n_segments,
                "error": err,
            }
        )
    return {
        "processed_root": str(processed_root.resolve()),
        "scenes": rows,
        "n_ready": sum(1 for r in rows if r["ready"]),
        "n_total": len(rows),
    }


def eval_mask_for_scene(metadata: pd.DataFrame, scene_id: str) -> np.ndarray:
    """Boolean mask of rows valid for cascade scoring on this scene."""
    forbidden = FORBIDDEN_GLOBAL_LABELS.get(scene_id, frozenset())
    if not forbidden:
        return np.ones(len(metadata), dtype=bool)
    if "global_label" not in metadata.columns:
        raise ValueError("metadata missing global_label column")
    labels = metadata["global_label"].astype(str)
    return ~labels.isin(forbidden).to_numpy()


def all_scene_indices(metadata: pd.DataFrame) -> np.ndarray:
    """All row indices (pool all sensor nodes within the scene)."""
    return np.arange(len(metadata), dtype=np.int64)


def build_stratified_holdout_indices(
    metadata: pd.DataFrame,
    scene_id: str,
    *,
    eval_fraction: float = HOLDOUT_EVAL_FRACTION,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split scene rows into eval holdout vs reserve (stratified by global_label)."""
    eval_mask = eval_mask_for_scene(metadata, scene_id)
    pool = np.where(eval_mask)[0]
    if len(pool) == 0:
        raise ValueError(f"No eval-eligible rows for scene {scene_id}")

    labels = metadata.iloc[pool]["global_label"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    eval_idx: list[int] = []
    reserve_idx: list[int] = []

    for label in np.unique(labels):
        label_indices = pool[labels == label]
        rng.shuffle(label_indices)
        n_eval = max(1, int(round(len(label_indices) * eval_fraction)))
        eval_idx.extend(label_indices[:n_eval].tolist())
        reserve_idx.extend(label_indices[n_eval:].tolist())

    return np.sort(np.array(eval_idx, dtype=np.int64)), np.sort(
        np.array(reserve_idx, dtype=np.int64)
    )


def save_holdout_artifacts(
    metadata: pd.DataFrame,
    scene_id: str,
    eval_idx: np.ndarray,
    reserve_idx: np.ndarray,
    out_dir: Path,
) -> dict[str, Path]:
    """Write holdout indices + tagged manifest for Person B."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = out_dir / f"scene_detector_{scene_id}_holdout_eval_indices.npy"
    reserve_path = out_dir / f"scene_detector_{scene_id}_holdout_reserve_indices.npy"
    np.save(eval_path, eval_idx)
    np.save(reserve_path, reserve_idx)

    manifest_rows = []
    for split_name, indices in (("eval", eval_idx), ("reserve", reserve_idx)):
        for idx in indices:
            row = metadata.iloc[int(idx)]
            manifest_rows.append(
                {
                    "index": int(idx),
                    "split": split_name,
                    "scene_id": scene_id,
                    "sensor_id": str(row.get("sensor_id", "")),
                    "global_label": str(row.get("global_label", "")),
                    "run_id": str(row.get("run_id", "")),
                }
            )
    manifest_path = out_dir / f"scene_detector_{scene_id}_holdout_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)

    meta_json = out_dir / f"scene_detector_{scene_id}_holdout_meta.json"
    meta_json.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "eval_fraction": HOLDOUT_EVAL_FRACTION,
                "n_eval": int(len(eval_idx)),
                "n_reserve": int(len(reserve_idx)),
                "forbidden_labels": sorted(FORBIDDEN_GLOBAL_LABELS.get(scene_id, [])),
                "eval_indices_path": str(eval_path),
                "reserve_indices_path": str(reserve_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "eval_indices": eval_path,
        "reserve_indices": reserve_path,
        "manifest": manifest_path,
        "meta": meta_json,
    }


def load_holdout_eval_indices(
    scene_id: str,
    checkpoints_dir: Path | str = "checkpoints",
) -> np.ndarray:
    path = Path(checkpoints_dir) / f"scene_detector_{scene_id}_holdout_eval_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Holdout not found: {path}. Run profile_scene_detector_prep.py --holdout."
        )
    return np.load(path)
