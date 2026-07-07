"""Unit tests for confidence-profile scene detector (Person B Day 1)."""

from __future__ import annotations

import numpy as np

from utils.confidence_scene_detector import (
    ConfidenceSceneDetector,
    KiWindowStats,
    SceneProfileBank,
    make_toy_profile_bank,
)


def _fill_window(detector: ConfidenceSceneDetector, vec: np.ndarray, n: int) -> None:
    for _ in range(n):
        stats = KiWindowStats()
        for i, ki in enumerate(["K0", "K1", "K2", "K3", "K4", "K5", "K6"]):
            stats.confidences[ki] = float(vec[i * 2])
            stats.accepted[ki] = bool(vec[i * 2 + 1] > 0.5)
        detector.update(stats)


def test_toy_bank_nearest_scene() -> None:
    bank = make_toy_profile_bank()
    target = bank.profiles[bank.scene_ids.index("i22")]
    _scene, dist, _ = bank.nearest(target)
    assert _scene == "i22"
    assert dist < 1e-6


def test_detector_hysteresis_requires_k_wins() -> None:
    bank = make_toy_profile_bank()
    detector = ConfidenceSceneDetector(
        bank=bank,
        window_n=4,
        check_every_m=1,
        hysteresis_k=3,
        default_scene_id="h24",
    )
    i22_vec = bank.profiles[bank.scene_ids.index("i22")]

    def one_update() -> None:
        stats = KiWindowStats()
        for i, ki in enumerate(["K0", "K1", "K2", "K3", "K4", "K5", "K6"]):
            stats.confidences[ki] = float(i22_vec[i * 2])
            stats.accepted[ki] = bool(i22_vec[i * 2 + 1] > 0.5)
        detector.update(stats)

    # Updates 1–3: fill window. Update 4: first scoring check (wins=1).
    for _ in range(4):
        one_update()
    assert detector.active_scene_id == "h24"
    one_update()  # 2nd check
    assert detector.active_scene_id == "h24"
    one_update()  # 3rd check → swap
    assert detector.active_scene_id == "i22"
    assert detector.swap_count == 1


def test_profile_bank_from_roundtrip_dict(tmp_path) -> None:
    bank = make_toy_profile_bank(["h08", "h24"])
    payload = {
        "profiles": {
            sid: {"vector": bank.profiles[i].tolist()}
            for i, sid in enumerate(bank.scene_ids)
        },
        "normalization": {
            "z_mean": bank.z_mean.tolist(),
            "z_std": bank.z_std.tolist(),
        },
        "feature_names": bank.feature_names,
    }
    path = tmp_path / "profiles.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = SceneProfileBank.from_json(path)
    assert loaded.scene_ids == ["h08", "h24"]
    assert loaded.profiles.shape == (2, 14)
