"""Confidence-profile scene detector (runtime): sliding window + hysteresis.

Detects M3N-VC scene_id (h24, h08, …) from cascade confidence statistics.
No separate model on raw audio/seismic input.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.scene_calibration import IDK_KI_NAMES
from utils.scene_profiles import FEATURE_NAMES

# Defaults — lock with docs/scene_detector_eval_protocol.md
DEFAULT_WINDOW_N = 64
DEFAULT_CHECK_EVERY_M = 16
DEFAULT_HYSTERESIS_K = 3


@dataclass(frozen=True)
class SceneProfileBank:
    """Reference 14-dim fingerprints per M3N-VC scene."""

    scene_ids: list[str]
    profiles: np.ndarray  # (n_scenes, n_features)
    feature_names: list[str]
    z_mean: np.ndarray
    z_std: np.ndarray

    @classmethod
    def from_json(cls, path: Path | str) -> SceneProfileBank:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        scene_ids = sorted(raw["profiles"].keys())
        profiles = np.array(
            [raw["profiles"][s]["vector"] for s in scene_ids], dtype=np.float64,
        )
        norm = raw.get("normalization", {})
        z_mean = np.array(norm.get("z_mean", profiles.mean(axis=0)), dtype=np.float64)
        z_std = np.array(norm.get("z_std", profiles.std(axis=0)), dtype=np.float64)
        z_std = np.where(z_std < 1e-9, 1.0, z_std)
        return cls(
            scene_ids=scene_ids,
            profiles=profiles,
            feature_names=list(raw.get("feature_names", FEATURE_NAMES)),
            z_mean=z_mean,
            z_std=z_std,
        )

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.z_mean) / self.z_std

    def distances(self, window_vec: np.ndarray) -> dict[str, float]:
        """Euclidean distance from window vector to each scene profile."""
        w = self._normalize(np.asarray(window_vec, dtype=np.float64))
        out: dict[str, float] = {}
        for i, scene_id in enumerate(self.scene_ids):
            p = self._normalize(self.profiles[i])
            out[scene_id] = float(np.linalg.norm(w - p))
        return out

    def nearest(self, window_vec: np.ndarray) -> tuple[str, float, dict[str, float]]:
        dists = self.distances(window_vec)
        scene_id = min(dists, key=dists.get)
        return scene_id, dists[scene_id], dists


@dataclass
class KiWindowStats:
    """Per-sample Ki confidence stats for one segment."""

    confidences: dict[str, float] = field(default_factory=dict)
    accepted: dict[str, bool] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Aggregate to 14-dim vector (mean conf + accept rate per Ki)."""
        values: list[float] = []
        for ki in IDK_KI_NAMES:
            if ki in self.confidences:
                values.append(float(self.confidences[ki]))
                values.append(float(self.accepted.get(ki, False)))
            else:
                values.extend([0.0, 0.0])
        return np.array(values, dtype=np.float64)


@dataclass
class ConfidenceSceneDetector:
    """Sliding-window profile matcher with hysteresis before threshold swap."""

    bank: SceneProfileBank
    window_n: int = DEFAULT_WINDOW_N
    check_every_m: int = DEFAULT_CHECK_EVERY_M
    hysteresis_k: int = DEFAULT_HYSTERESIS_K
    default_scene_id: str | None = None

    _buffer: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _samples_since_check: int = field(default=0, init=False, repr=False)
    _active_scene_id: str | None = field(default=None, init=False, repr=False)
    _candidate_scene_id: str | None = field(default=None, init=False, repr=False)
    _consecutive_wins: int = field(default=0, init=False, repr=False)
    _last_candidate: str | None = field(default=None, init=False, repr=False)
    _swap_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.default_scene_id is None:
            self.default_scene_id = self.bank.scene_ids[0]
        self._active_scene_id = self.default_scene_id

    @property
    def active_scene_id(self) -> str:
        assert self._active_scene_id is not None
        return self._active_scene_id

    @property
    def swap_count(self) -> int:
        return self._swap_count

    def reset(self) -> None:
        self._buffer.clear()
        self._samples_since_check = 0
        self._active_scene_id = self.default_scene_id
        self._candidate_scene_id = None
        self._consecutive_wins = 0
        self._last_candidate = None
        self._swap_count = 0

    def update(self, stats: KiWindowStats) -> None:
        """Push one sample's Ki stats; may update candidate/active scene."""
        self._buffer.append(stats.to_feature_vector())
        while len(self._buffer) > self.window_n:
            self._buffer.popleft()

        if len(self._buffer) < self.window_n:
            return

        self._samples_since_check += 1
        if self._samples_since_check < self.check_every_m:
            return
        self._samples_since_check = 0

        window_vec = np.mean(np.stack(list(self._buffer), axis=0), axis=0)
        candidate, _dist, _all = self.bank.nearest(window_vec)
        self._apply_hysteresis(candidate)

    def _apply_hysteresis(self, candidate: str) -> None:
        if candidate == self._candidate_scene_id:
            self._consecutive_wins += 1
        else:
            self._candidate_scene_id = candidate
            self._consecutive_wins = 1

        if (
            self._consecutive_wins >= self.hysteresis_k
            and candidate != self._active_scene_id
        ):
            self._active_scene_id = candidate
            self._swap_count += 1

        self._last_candidate = candidate

    def candidate_scene_id(self) -> str | None:
        return self._last_candidate

    def window_ready(self) -> bool:
        return len(self._buffer) >= self.window_n


def make_toy_profile_bank(
    scene_ids: list[str] | None = None,
    *,
    seed: int = 0,
) -> SceneProfileBank:
    """Synthetic profiles for unit tests (Person B Day 1)."""
    scene_ids = scene_ids or ["h08", "h24", "s31", "a06", "i29", "i22"]
    rng = np.random.default_rng(seed)
    profiles = rng.uniform(0.5, 0.95, size=(len(scene_ids), 14))
    # Make each scene unique along diagonal features for separability.
    for i, _sid in enumerate(scene_ids):
        profiles[i, i % 14] += 0.5
    z_mean = profiles.mean(axis=0)
    z_std = profiles.std(axis=0)
    z_std = np.where(z_std < 1e-9, 1.0, z_std)
    return SceneProfileBank(
        scene_ids=scene_ids,
        profiles=profiles,
        feature_names=FEATURE_NAMES.copy(),
        z_mean=z_mean,
        z_std=z_std,
    )
