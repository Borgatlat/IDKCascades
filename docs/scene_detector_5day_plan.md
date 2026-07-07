# Scene Detector — 5-Day Plan (2 People, Corrected)

**Scenes = M3N-VC environments:** `h08`, `h24`, `s31`, `a06`, `i29`, `i22` — **not** sensor nodes (`rs1`, `rs2`, …).

**Reference:** `docs/scene_detector_plan.md`, `checkpoints/scene_catalog.json`

**Default experiment split (lock Day 1):**

| Role | Scenes |
|------|--------|
| Calibration / profile build | `h08`, `h24`, `s31`, `a06`, `i29` (pool all sensors per scene) |
| Deployment holdout (eval) | `i22` (concrete, sunny; **no gle350**) |

Change only if data or optimizer is not ready — document any change in team chat.

**Hyperparameters (lock Day 1):** N=64, M=16, K=3, profile = 14-dim (7 Ki × mean conf + accept rate).

**Friday must-ship:** `scene_detector_comparison.json` + one figure (B vs Oracle vs Detector on `i22` holdout).

---

## Roles

| | **Person A** | **Person B** |
|---|--------------|--------------|
| **Title** | Data & baselines | Detector & evaluation |
| **Owns** | Processed multi-scene data, profiles, B + oracle, traces, separability | `confidence_scene_detector.py`, hysteresis, full eval, figures |
| **Does not own** | Detector state machine | Threshold bank creation (optimizer team) — only consumes it |

---

## Day 1 (Monday) — Align & parallel start

### Person A

| Time | Task | Details |
|------|------|---------|
| AM | **Team sync (30 min)** | Confirm scene IDs, holdout=`i22`, N/M/K, label rules (a06: no mustang; i22: no gle350). |
| AM | **Data audit** | Open `checkpoints/scene_catalog.json`. For each scene, check `datasets/processed/<scene_id>/` exists (run `python process_scenes.py --scenes <id>` if missing). |
| PM | **Define eval protocol** | Write `docs/scene_detector_eval_protocol.md` (or section in team notes): holdout scene, tune fraction if any, metrics list, forbidden labels per scene. |
| PM | **Holdout indices v0** | From `i22` metadata: stratified sample indices (e.g. 80% eval pool), save `checkpoints/scene_detector_i22_holdout_indices.npy`. Tag each index with `scene_id=i22` in a small parquet/CSV. |
| PM | **Request from optimizer team** | JSON schema for `scene_threshold_bank.json` and `scene_confidence_profiles.json`; ETA per scene. |

**Day 1 deliverables (A):** Protocol doc, holdout index file, data-readiness checklist per scene.

### Person B

| Time | Task | Details |
|------|------|---------|
| AM | **Same sync** | Agree file names and handoff format for traces. |
| AM | **Read code** | `cascade/inference.py`, `utils/classifier_registry.py` (`with_threshold_overrides`), skim `utils/scene_calibration.py`. |
| PM | **Create module skeleton** | `utils/confidence_scene_detector.py`: classes `SceneProfile`, `ConfidenceSceneDetector` with stubs: `update()`, `candidate_scene()`, `active_scene()`. |
| PM | **Toy unit tests** | 6 fake scenes × 14-dim profiles; assert nearest-scene picks correct fake ID; no real data yet. |

**Day 1 deliverables (B):** Skeleton module + passing toy tests.

**End of Day 1:** Both aligned. B does **not** block on A.

---

## Day 2 (Tuesday) — Profiles + window logic

### Person A

| Time | Task | Details |
|------|------|---------|
| AM | **Draft confidence profiles** | If optimizer profiles not ready: per `scene_id`, run cascade (or per-Ki forward) on **all sensors pooled**, compute mean conf + accept rate per K0–K6 → 14-dim vector. Save `checkpoints/scene_confidence_profiles.json`. |
| PM | **Profile separability** | 6×6 distance matrix (h08…i22). Confirm `i22` row is closest to itself vs others. If not separable, flag team before Day 3. |
| PM | **Condition B (fixed thresholds)** | On `i22` holdout: single global \(H_i\) from registry → cascade eval. Save `checkpoints/scene_detector_baseline_fixed.json`. |
| PM | **Handoff to B** | Send: `scene_confidence_profiles.json`, holdout indices, B baseline JSON (partial OK). |

**Day 2 deliverables (A):** Profiles JSON, separability table, B baseline, handoff message to B.

### Person B

| Time | Task | Details |
|------|------|---------|
| AM | **Sliding window** | `deque` maxlen=N; on each sample append per-Ki (conf, accepted); compute window mean → 14-dim vector. |
| AM | **Normalization** | Z-score using means/stds from **reference profiles** (not live window). |
| PM | **Nearest scene** | Euclidean distance to each `scene_id` in profile bank; return argmin as `candidate_scene_id`. |
| PM | **Plug A’s profiles** | Replace toy profiles with real JSON; run offline on **synthetic trace** (random confs) then wait for real traces Day 3. |

**Day 2 deliverables (B):** Window + distance working on real profile file format.

**Handoff:** A → B sends profiles + holdout indices by **EOD Tuesday**.

---

## Day 3 (Wednesday) — Oracle + traces (main handoff)

### Person A

| Time | Task | Details |
|------|------|---------|
| AM | **Load threshold bank** | From optimizer: `scene_threshold_bank.json`. Fallback: registry defaults + any per-scene overrides available. |
| AM | **Condition Oracle** | On `i22` holdout: always use `threshold_bank["i22"]` (true scene). Cascade eval with **gle350 segments excluded** from scoring. Save `checkpoints/scene_detector_oracle_eval.json`. |
| PM | **Export confidence traces** | Per holdout sample: `idx`, `scene_id`, for each Ki fired: `confidence`, `accepted`, `threshold_hi`. Save `checkpoints/scene_detector_confidence_traces.parquet`. |
| PM | **Targets sheet** | One table: B vs Oracle — accuracy, \(K_{\mathrm{det}}\), \(\bar{C}\), per-Ki \(P(\mathrm{IDK})\). Send to B. |

**Day 3 deliverables (A):** Oracle JSON, confidence traces, targets sheet. **This unblocks B’s real replay.**

### Person B

| Time | Task | Details |
|------|------|---------|
| AM | **Offline replay** | Feed traces into detector; every M samples compute candidate; measure **scene detection accuracy** (% windows where candidate == `i22`). |
| PM | **Hysteresis** | State: `active_scene_id`, `candidate`, `consecutive_wins`. Swap only after K wins. Unit test: oscillating candidates → no swap until K stable. |
| PM | **Detection report** | Save `checkpoints/scene_detector_offline_accuracy.json` (accuracy, confusion vs all 6 scenes). |

**Day 3 deliverables (B):** Hysteresis tested; offline detection accuracy on i22 traces.

**Gate:** Detection accuracy on i22 should beat random (1/6 ≈ 17%). Target >70% after warm-up.

---

## Day 4 (Thursday) — Integration + detector eval

### Person A

| Time | Task | Details |
|------|------|---------|
| AM | **Sanity-check B** | Verify B and oracle used **identical** holdout indices and same label filter (no gle350 on i22). |
| AM | **Label filter helper** | Small util: `eval_mask_for_scene(metadata, scene_id)` — documents a06/i22 exclusions for B’s scripts. |
| PM | **Support B** | If bank files update from optimizer, refresh profiles/oracle only if indices unchanged; else re-run oracle only. |
| PM | **Draft methods** | Half-page: scenes defined, 14-dim profile, not sensor-based. |

**Day 4 deliverables (A):** Verified index alignment; label filter documented; methods draft.

### Person B

| Time | Task | Details |
|------|------|---------|
| AM | **Threshold swap wiring** | On `active_scene_id` change: `registry.with_threshold_overrides(threshold_bank[active_scene_id])`. |
| AM | **`profile_confidence_scene_detector.py`** | CLI: load bank, profiles, traces; simulate streaming; run cascade eval for **Condition Detector**. |
| PM | **Full Detector run** | i22 holdout with detector-selected thresholds + hysteresis. Log swap count. |
| PM | **Draft comparison table** | Rows: B, Oracle, Detector — same metrics as targets sheet. |

**Day 4 deliverables (B):** End-to-end detector eval; draft `scene_detector_comparison.json`.

**Gate:** Detector should beat B on \(K_{\mathrm{det}}\) and/or \(\bar{C}\), or team debugs Friday AM.

---

## Day 5 (Friday) — Finalize, figures, demo

### Person A

| Time | Task | Details |
|------|------|---------|
| AM | **Verify final numbers** | Reconcile B and oracle rows in comparison JSON. |
| AM | **Optional ablation data** | If B has time: run oracle on second holdout scene (e.g. `a06`) with mustang excluded — stretch goal. |
| PM | **Results paragraph** | 3–5 sentences: fixed vs oracle vs detector on i22. |
| PM | **Demo prep** | With B: 5-min script for team/advisor. |

**Day 5 deliverables (A):** Results text, verified table.

### Person B

| Time | Task | Details |
|------|------|---------|
| AM | **Fix bugs** | If detector < B: check hysteresis, profile normalization, wrong bank row. |
| AM | **Optional ablation** | K=1 vs K=3 or N=32 vs N=64 — one row in comparison. |
| PM | **Figures** | `checkpoints/figures/scene_detector/`: (1) bar chart B/Oracle/Detector, (2) confusion matrix over **scenes** not sensors. |
| PM | **Freeze artifacts** | Final `scene_detector_comparison.json` + figure paths in JSON. |

**Day 5 deliverables (B):** Final JSON, PNGs, swap count in summary.

### Both (Friday 4 PM, 30 min)

- [ ] Comparison table complete  
- [ ] One figure ready  
- [ ] Can explain: scenes = h24/h08/…, not rs1/rs2  
- [ ] Can explain hysteresis in one sentence  
- [ ] Limitations noted: single holdout scene week 1; a06/i22 label constraints  

---

## Handoff checklist (Person A → Person B)

| Item | Due |
|------|-----|
| Holdout indices (`i22`) | Day 1 EOD |
| `scene_confidence_profiles.json` | Day 2 EOD |
| B baseline JSON | Day 2 EOD |
| Confidence traces parquet | Day 3 noon |
| Oracle JSON + targets sheet | Day 3 EOD |
| Threshold bank (from optimizer or fallback) | Day 3 AM |

---

## Dependencies on optimizer team

| Need | Used for | Fallback |
|------|----------|----------|
| `scene_threshold_bank.json` | Oracle + Detector | Global registry \(H_i\) for all scenes (weak) |
| Per-scene optimized \(H_i\) | Fair comparison | Person A draft tune per scene (Sub-step C style) |

Person A should ping optimizer daily Days 1–3.

---

## Files created by end of week

```
docs/scene_detector_eval_protocol.md          (A, Day 1)
checkpoints/scene_confidence_profiles.json    (A, Day 2)
checkpoints/scene_detector_baseline_fixed.json
checkpoints/scene_detector_oracle_eval.json
checkpoints/scene_detector_confidence_traces.parquet
checkpoints/scene_detector_offline_accuracy.json  (B, Day 3)
checkpoints/scene_detector_comparison.json        (B, Day 5)
checkpoints/figures/scene_detector/*.png          (B, Day 5)
utils/confidence_scene_detector.py                (B)
profile_confidence_scene_detector.py              (B)
```

---

## What we are NOT doing this week

- Sensor node classification (`rs1` vs `rs2`)  
- Raw-audio / spectrogram scene classifier  
- Retraining Ki weights  
- Full 6-scene streaming crossover eval (stretch: Day 5 optional second holdout)

---

## One-line daily standup prompts

| Day | Person A | Person B |
|-----|----------|----------|
| 1 | "Is all scene data processed?" | "Do toy tests pass?" |
| 2 | "Are profiles separable? B baseline done?" | "Does window match profile format?" |
| 3 | "Traces + oracle handed off?" | "Detection accuracy on i22?" |
| 4 | "Indices match? Methods drafted?" | "Detector end-to-end runs?" |
| 5 | "Numbers verified? Demo ready?" | "Figure + final JSON frozen?" |
