# Scene Detector — Eval Protocol

**Scenes** = M3N-VC environments: `h08`, `h24`, `s31`, `a06`, `i29`, `i22` (terrain + weather).  
**Not scenes:** sensor nodes `rs1`, `rs2`, … (pool all nodes within a scene).

## Default split

| Role | Scenes |
|------|--------|
| Calibration / profile build | `h08`, `h24`, `s31`, `a06`, `i29` |
| Deployment holdout (eval) | `i22` |

## Label constraints (scoring)

| Scene | Excluded global labels |
|-------|------------------------|
| `a06` | `mustang` |
| `i22` | `gle350` |

Apply via `utils.scene_data.eval_mask_for_scene()` before accuracy / holdout splits.

## Holdout

- **Fraction:** 80% of eligible rows → eval holdout; 20% reserve  
- **Stratified by:** `global_label`  
- **Artifacts:** `checkpoints/scene_detector_<scene>_holdout_eval_indices.npy`

## Detector hyperparameters

| Symbol | Value |
|--------|-------|
| N (window) | 64 |
| M (re-check) | 16 |
| K (hysteresis) | 3 |
| Profile dim | 14 (7 Ki × mean confidence + accept rate) |
| Distance | Euclidean on z-scored profiles |

## Experiment conditions

| ID | Thresholds | Scene ID |
|----|------------|----------|
| B | Fixed global registry \(H_i\) | Ignored |
| Oracle | `threshold_bank[true scene_id]` | Ground truth |
| Detector | `threshold_bank[active_scene_id]` | Confidence window + hysteresis |

## Metrics (all conditions)

- End-to-end cascade accuracy (label-filtered)
- \(K_{\mathrm{det}}\) rate
- Mean latency \(\bar{C}\), p95
- Per-Ki \(P(\mathrm{IDK})\) (optional)
- Scene detection accuracy (% windows correct) — Detector only
- Threshold swap count — Detector only

## Optimizer team file contracts

See `checkpoints/scene_threshold_bank_schema.json`.

## Commands (Person A)

```bash
python profile_scene_detector_prep.py --audit
python profile_scene_detector_prep.py --holdout --deployment-scene i22
python profile_scene_detector_prep.py --profiles
python profile_scene_detector_prep.py --baseline-b --deployment-scene i22
```

## Commands (Person B)

```bash
python -m pytest tests/test_confidence_scene_detector.py -v
# After profiles exist:
python -c "from utils.confidence_scene_detector import SceneProfileBank; print(SceneProfileBank.from_json('checkpoints/scene_confidence_profiles.json').scene_ids)"
```
