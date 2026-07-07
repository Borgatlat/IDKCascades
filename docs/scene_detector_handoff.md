# Teammate Handoff — Scene Detector (Person B)

## Start here

**Start here:** `docs/team_role_pick_5day.md` — **remaining work only** (skips what is already built).

1. Read `docs/scene_detector_plan.md` and `docs/scene_detector_eval_protocol.md`
2. Run tests: `python -m pytest tests/test_confidence_scene_detector.py -v`
3. Open `utils/confidence_scene_detector.py` — your main module

## What Person A delivered (Day 1–2)

| File | Purpose |
|------|---------|
| `checkpoints/scene_data_readiness.json` | Which scenes are preprocessed |
| `checkpoints/scene_confidence_profiles.json` | 14-dim reference vectors per scene |
| `checkpoints/scene_profile_separability.csv` | 6×6 distance matrix |
| `checkpoints/scene_detector_i22_holdout_*` | Eval indices (when i22 processed) |
| `checkpoints/scene_detector_baseline_fixed.json` | Condition B (when i22 ready) |

**Note:** If `i22` is still preprocessing, profiles may only include `h08`, `h24`, `i29` for now. Use toy bank for logic until traces arrive Day 3.

## Your tasks (remaining — core module already done)

- [x] `utils/confidence_scene_detector.py` (window, hysteresis, profile bank)
- [x] Toy tests in `tests/test_confidence_scene_detector.py`
- [ ] Smoke test with real `scene_confidence_profiles.json`
- [ ] Offline replay on traces (Day 3, from Person A)
- [ ] Create `profile_confidence_scene_detector.py` + Condition Detector eval
- [ ] Figures + final `scene_detector_comparison.json`

See `docs/team_role_pick_5day.md` for full Role B checklist.

## Load real profiles

```python
from pathlib import Path
from utils.confidence_scene_detector import SceneProfileBank, ConfidenceSceneDetector, KiWindowStats

bank = SceneProfileBank.from_json("checkpoints/scene_confidence_profiles.json")
detector = ConfidenceSceneDetector(bank=bank, default_scene_id="h24")

stats = KiWindowStats(
    confidences={"K0": 0.91, "K1": 0.88},
    accepted={"K0": True, "K1": False},
)
detector.update(stats)
print(detector.candidate_scene_id(), detector.active_scene_id)
```

## Profile JSON shape

```json
{
  "profiles": {
    "h24": {
      "scene_id": "h24",
      "vector": [14 floats],
      "feature_names": ["K0_mean_confidence", "K0_accept_rate", ...]
    }
  },
  "normalization": { "z_mean": [...], "z_std": [...] }
}
```

## Do not

- Classify `sensor_id` (rs1 vs rs2) — wrong problem
- Wait for all 6 scenes to start coding — use toy tests + partial profiles

## Questions → Person A

- Holdout indices path when i22 lands
- Oracle numbers for comparison (Day 3)
