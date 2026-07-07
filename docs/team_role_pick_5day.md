# Pick Your Role — 5-Day Scene Detector Sprint (Remaining Work Only)

**Either teammate picks Role A or Role B.** Skip everything in **Already done** — that is inherited from this week.

> **Friday must-ship (both):** `checkpoints/scene_detector_comparison.json` + one PNG (**B vs Oracle vs Detector** on `i22`).

---

## Already done — do NOT redo

| Item | Location |
|------|----------|
| All 6 scenes preprocessed | `datasets/processed/` |
| Streaming parquet loader + memmap | `process_data.py`, `process_scenes.py` |
| Scene catalog | `checkpoints/scene_catalog.json` |
| Eval protocol | `docs/scene_detector_eval_protocol.md` |
| Draft confidence profiles (3-scene partial possible) | `checkpoints/scene_confidence_profiles.json` |
| Draft separability matrix | `checkpoints/scene_profile_separability.csv` |
| Threshold bank **schema** | `checkpoints/scene_threshold_bank_schema.json` |
| **Full detector core** (window N=64, M=16, K=3, hysteresis, profile bank) | `utils/confidence_scene_detector.py` |
| Toy unit tests (passing) | `tests/test_confidence_scene_detector.py` |
| Prep script | `profile_scene_detector_prep.py` |
| Ki weights K0–K6 + Kdet | `checkpoints/` + `TEAMMATE_SETUP.md` |
| GitHub branch pushed | `cursor/hierarchical-ki-training-pipeline` |

**Lock on Day 1 sync (30 min):** holdout=`i22`, N=64, M=16, K=3, desktop GPU timing, i22 excludes `gle350`.

---

## Role A — **Baselines & Data** (remaining only)

**Pick this if you like:** eval scripts, JSON artifacts, cascade metrics, results writing.

### Day 1
- [ ] **A1** Clone + `python verify_classifiers.py` → all `[OK]`
- [ ] **A2** `python profile_scene_detector_prep.py --audit` → fix until **6/6** in `scene_data_readiness.json`
- [ ] **A3** `python profile_scene_detector_prep.py --holdout --deployment-scene i22`
- [ ] **A4** Confirm `checkpoints/scene_detector_i22_holdout_eval_indices.npy` exists → message Role B the path

**Done when:** holdout indices exist; Role B can load them.

---

### Day 2
- [ ] **A5** **Re-run** profiles on all 6 scenes (draft may be partial):  
  `python profile_scene_detector_prep.py --profiles --max-samples 500`
- [ ] **A6** **Re-run** separability; confirm **i22** nearest to itself:  
  `python profile_scene_detector_prep.py --separability`
- [ ] **A7** Condition **B** (fixed global H_i on i22 holdout):  
  `python profile_scene_detector_prep.py --baseline-b --deployment-scene i22 --max-samples 500`
- [ ] **A8** Send Role B: `scene_confidence_profiles.json` + `scene_detector_baseline_fixed.json`

**Done when:** B baseline JSON has accuracy, Kdet rate, mean latency (ms).

---

### Day 3 — critical handoff
- [ ] **A9** Build or load `checkpoints/scene_threshold_bank.json`  
  (fallback: copy global registry H_i into every scene key)
- [ ] **A10** Condition **Oracle** on i22 → `scene_detector_oracle_eval.json`
- [ ] **A11** Export `scene_detector_confidence_traces.parquet` (`idx`, `scene_id`, per-Ki conf/accepted/threshold)
- [ ] **A12** Send Role B targets sheet: B vs Oracle (accuracy, Kdet, latency ms)

**Done when:** traces parquet + oracle JSON exist.

---

### Day 4
- [ ] **A13** Confirm Role B uses same holdout `.npy` + `eval_mask_for_scene(..., "i22")`
- [ ] **A14** Draft ½-page methods (scenes = h08…i22, 14-dim profile, not sensor nodes)
- [ ] **A15** Help debug if Detector loses to B (wrong threshold bank row)

---

### Day 5
- [ ] **A16** Verify B + Oracle rows in `scene_detector_comparison.json`
- [ ] **A17** Write 3–5 sentence results paragraph
- [ ] **A18** Joint 5-min demo with Role B

---

## Role B — **Integration & Eval** (remaining only)

**Pick this if you like:** wiring scripts, replay/eval loops, plots.

**Do NOT re-implement** window, hysteresis, or `SceneProfileBank` — already in `confidence_scene_detector.py`.

### Day 1
- [ ] **B1** Clone + `python -m pytest tests/test_confidence_scene_detector.py -v` → all pass
- [ ] **B2** Smoke-test **real** profiles (not toy bank):

```python
from utils.confidence_scene_detector import SceneProfileBank, ConfidenceSceneDetector, KiWindowStats
bank = SceneProfileBank.from_json("checkpoints/scene_confidence_profiles.json")
det = ConfidenceSceneDetector(bank=bank, default_scene_id="h24")
# feed ~64 updates; print det.candidate_scene_id(), det.active_scene_id
```

- [ ] **B3** Read `cascade/inference.py` + `utils/classifier_registry.py` (`with_threshold_overrides`)
- [ ] **B4** Add **one** test: load real `scene_confidence_profiles.json` and assert 6 scene IDs (optional stretch)

**Done when:** real JSON loads; tests still pass.

---

### Day 2
- [ ] **B5** *(Light day)* Review Role A’s refreshed profiles + separability CSV when A finishes A5–A6
- [ ] **B6** If profiles changed: re-run smoke test from B2
- [ ] **B7** Sketch `profile_confidence_scene_detector.py` CLI args (bank, profiles, holdout indices, threshold bank)

**Done when:** CLI outline agreed; ready for traces Day 3.

---

### Day 3
- [ ] **B8** Load `scene_detector_confidence_traces.parquet` (from Role A)
- [ ] **B9** Offline replay through `ConfidenceSceneDetector` (no re-inference)
- [ ] **B10** Scene detection accuracy on i22 (target **>70%** after warm-up N=64)
- [ ] **B11** Save `scene_detector_offline_accuracy.json` + scene confusion table

**Done when:** offline accuracy beats random (16.7%).

---

### Day 4
- [ ] **B12** **Create** `profile_confidence_scene_detector.py` (does not exist yet)
- [ ] **B13** On scene swap: `registry.with_threshold_overrides(threshold_bank[active_scene_id])`
- [ ] **B14** Run Condition **Detector** on i22 holdout (desktop CUDA sync for latency)
- [ ] **B15** Draft `scene_detector_comparison.json` — rows: **B**, **Oracle**, **Detector**

**Done when:** all 3 rows in comparison JSON.

---

### Day 5
- [ ] **B16** Debug if Detector < B (normalization, bank row, hysteresis K)
- [ ] **B17** Plot `checkpoints/figures/scene_detector/comparison_bar.png`
- [ ] **B18** Plot `checkpoints/figures/scene_detector/scene_confusion.png`
- [ ] **B19** Freeze final JSON + figure paths; include `swap_count`

---

## Handoff (only what is still missing)

| From | To | Due | Artifact | Status |
|------|-----|-----|----------|--------|
| A → B | Holdout indices | Day 1 | `scene_detector_i22_holdout_eval_indices.npy` | **TODO** |
| A → B | Refreshed profiles + B baseline | Day 2 | `scene_confidence_profiles.json`, `scene_detector_baseline_fixed.json` | profiles draft exists; **B baseline TODO** |
| A → B | Traces + Oracle | Day 3 | `scene_detector_confidence_traces.parquet`, `scene_detector_oracle_eval.json` | **TODO** |
| B → A | Comparison draft | Day 4 | `scene_detector_comparison.json` | **TODO** |

---

## How to choose

| **Role A** | **Role B** |
|------------|------------|
| Run prep/eval scripts | Wire detector into cascade eval |
| Own B + Oracle numbers | Own Detector + figures |
| Unblocks B on Day 3 | Starts Day 1 with existing module + tests |

**Rule:** one picks A, the other gets B. Decide at Day 1 sync.

---

## Daily standup (remaining milestones only)

| Day | A | B |
|-----|---|---|
| 1 | Holdout saved? | Real profiles smoke test OK? |
| 2 | B baseline JSON sent? | CLI sketched? |
| 3 | Traces + Oracle sent? | Offline accuracy on i22? |
| 4 | Methods draft? | End-to-end Detector runs? |
| 5 | Numbers verified? | Figure + JSON frozen? |

---

## Links

- Repo: https://github.com/Borgatlat/IDKCascades (branch `cursor/hierarchical-ki-training-pipeline`)
- Setup: `TEAMMATE_SETUP.md`
- Processed data: Drive zip of `datasets/processed/` (not on GitHub)
