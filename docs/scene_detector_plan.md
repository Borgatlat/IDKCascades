# Scene Detector Plan (M3N-VC) — Corrected

**Status:** Team spec (replaces earlier `sensor_id` = scene assumption)

---

## Critical correction

| Wrong (do not use for scene detector) | Correct |
|---------------------------------------|---------|
| Scene = sensor node (`rs1`, `rs2`, …) | Scene = **M3N-VC dataset environment** (`h24`, `h08`, `s31`, `a06`, `i29`, `i22`) |
| Detect which microphone recorded the clip | Detect **terrain + weather + acoustic environment** (domain) |
| Leave-one-**sensor**-out | Leave-one-**scene**-out (or calibration scenes vs deployment scene) |

**Why:** Differences between sensor nodes within one scene are **minuscule** compared to differences between scenes (asphalt vs dirt vs concrete; sunny vs rainy vs windy). The hierarchical paper’s calibration-maintenance problem is about **domain shift across deployment environments**, not node placement.

**Note on existing scripts:** `profile_scene_shift_baseline.py` / `utils/scene_inventory.py` used `sensor_id` as a **h24-only WIP proxy** to prototype drift metrics. That work is still useful for pipeline plumbing, but the **production scene detector** targets **scene IDs** below.

---

## What is a “scene”?

Each row is one M3N-VC collection environment (see `checkpoints/scene_catalog.json`, `process_data.KNOWN_SCENES`).

| Scene | Terrain | Weather | Target vehicles (paper codes) |
|-------|---------|---------|-------------------------------|
| **h08** | Asphalt & gravel | Sunny | C, G, M, X (cx30, gle350, mustang, miata) |
| **h24** | Asphalt & gravel | **Rainy** | C, G, M, X + background |
| **s31** | Dirt & gravel | Sunny | C, G, M, X + background |
| **a06** | Asphalt | Sunny | **C, G, X only — no Mustang (M)** |
| **i29** | Concrete | Windy | C, G, M, X |
| **i22** | Concrete | Sunny | **C, M, X only — no GLE350 (G)** |

**Label constraints for cross-scene eval (must respect):**

- **a06:** no **mustang** as a deployment target — do not score or route mustang labels on a06-held-out eval.
- **i22:** no **gle350** — do not score gle350 on i22-held-out eval.
- When building per-scene threshold banks and profiles, Ki label spaces and cascade branches may differ by scene; use scene-specific metadata from `datasets/processed/<scene_id>/`.

---

## Core mechanism (unchanged)

We are **not** training a separate raw-audio / seismic scene classifier.

1. **Offline (per scene):** Run the threshold optimizer on that scene → save optimized \(\{H_0,\ldots,H_6\}\) **and** a **14-dim reference profile** (7 Ki × mean confidence + accept rate). Profiles are a free byproduct of the optimizer’s outcomes pass.
2. **Runtime:** Buffer the last **N** samples’ cascade confidence statistics (numbers already computed every inference).
3. **Every M samples:** Average the window into a 14-dim vector; compare to each scene’s stored profile (e.g. Euclidean distance on z-scored vectors).
4. **Nearest match** = candidate scene.
5. **Hysteresis:** Require the candidate to win **K consecutive** windows (or by a margin) before swapping threshold rows — prevents threshold flapping at boundaries.
6. **Swap:** Load `threshold_bank[active_scene]` via `registry.with_threshold_overrides(...)`.

This detects the **same kind of shift** the paper cares about (confidence / calibration drift under domain change), with **zero extra forward passes**.

---

## Artifacts

| File | Owner | Contents |
|------|-------|----------|
| `checkpoints/scene_catalog.json` | Shared | Scene inventory (terrain, weather, labels, hours) — **exists** |
| `checkpoints/scene_threshold_bank.json` | Optimizer team | `{ "h24": {K0:…,…}, "h08": …, … }` |
| `checkpoints/scene_confidence_profiles.json` | Optimizer team (export) or Scene team (draft) | `{ "h24": [14 floats], … }` + normalization metadata |
| `checkpoints/scene_detector_comparison.json` | Scene team | B vs oracle vs detector metrics |
| `datasets/processed/<scene_id>/` | Data | Per-scene spectrograms + metadata (`process_scenes.py`) |

**Scene key everywhere:** `scene_id` ∈ `{h08, h24, s31, a06, i29, i22}` — never `sensor_id` for detection labels.

---

## Experiment conditions

| ID | Name | Thresholds | Scene selection |
|----|------|------------|-----------------|
| **B** | Fixed global | Single \(H_i\) set (calibration scene pool) | None |
| **Oracle** | Per-scene bank | `threshold_bank[true scene_id]` | Ground-truth scene (ceiling) |
| **Detector** | Per-scene bank | `threshold_bank[detected scene_id]` | Confidence window + hysteresis |
| **C** (optional) | Re-tune on deploy | Small labeled set on deployment scene | No detector — direct maintenance |

**Typical week-1 split:** Calibrate / optimize on `{h08, h24, s31, a06, i29}`; hold out **i22** (or another scene) as deployment. Adjust based on data readiness.

**Sensors within a scene:** Pool **all nodes** (rs1–rs8, etc.) for that scene’s profile and optimizer run. Do **not** treat nodes as separate detection classes.

---

## Hyperparameters (lock Day 1)

| Symbol | Meaning | Starting value |
|--------|---------|----------------|
| **N** | Sliding window length (segments) | 64 |
| **M** | Re-score every M new samples | 16 |
| **K** | Hysteresis: consecutive wins before swap | 3 |
| **δ** | Optional min distance margin to switch | TBD |
| **Profile dim** | 7 Ki × (mean conf, accept rate) | 14 |

---

## Two-person week plan

**→ Detailed 5-day schedule:** `docs/scene_detector_5day_plan.md`

### Person A — Data, profiles, baselines

| Day | Tasks |
|-----|--------|
| **1** | Sync on scene IDs + label constraints (a06/i22). Confirm processed data per scene (`process_scenes.py --all` or subset). Lock N/M/K. |
| **2** | Build **draft profiles** per scene if optimizer not ready (pool all sensors in scene). Run **Condition B** (fixed \(H_i\)) on deployment-scene holdout. Save holdout indices. |
| **3** | **Oracle** eval (true `scene_id` → bank row). Export **confidence traces** per sample (Ki, conf, accepted). **Profile separability** matrix (6×6 distances between h08…i22). |
| **4–5** | See 5-day plan: sanity-check, label filters, methods, demo. |

_Summary only — see `docs/scene_detector_5day_plan.md` for full day-by-day tasks._

**Handoff to Person B (by Day 3):** `holdout_indices`, `scene_confidence_profiles.json`, confidence traces, B + oracle JSON.

### Person B — Detector, hysteresis, eval, figures

| Day | Tasks |
|-----|--------|
| **1** | Stub `utils/confidence_scene_detector.py` + toy tests (fake 6 scenes × 14 dims). |
| **2** | Sliding window + z-score + Euclidean nearest **scene_id** (not sensor). |
| **3** | Plug real profiles; offline replay detection accuracy vs true `scene_id`. |
| **4** | Hysteresis state machine + threshold swap wiring + `profile_confidence_scene_detector.py`. |
| **5** | Full **Detector** condition; comparison table B / Oracle / Detector; figures. |

_Summary only — see `docs/scene_detector_5day_plan.md` for full day-by-day tasks._

---

## Code to read / reuse

| Area | Files |
|------|--------|
| Scene catalog & preprocess | `checkpoints/scene_catalog.json`, `process_scenes.py`, `process_data.KNOWN_SCENES` |
| Threshold swap | `utils/classifier_registry.py` → `with_threshold_overrides` |
| Confidence collection | `utils/scene_calibration.py` → `collect_ki_confidences`, `profile_all_kis` |
| Cascade eval | `cascade/eval.py`, `cascade/executor.py`, `cascade/inference.py` |
| **Deprecated for detector labels** | `utils/scene_inventory.py` `SCENE_PROXY = "sensor_id"` — h24 WIP only |

**New modules (to create):**

- `utils/confidence_scene_detector.py` — window, distance, hysteresis, `active_scene_id`
- `profile_confidence_scene_detector.py` — B / oracle / detector comparison

---

## Paper one-liner

> We extend per-scene threshold optimization across M3N-VC environments (h24 rainy asphalt, s31 dirt, i29 windy concrete, etc.) and maintain calibration at runtime by matching sliding-window cascade confidence profiles to stored scene fingerprints — without a dedicated scene classifier on raw input and without re-running the optimizer on every deploy.

---

## Glossary (team)

| Term | Meaning |
|------|---------|
| **scene_id** | `h24`, `h08`, `s31`, `a06`, `i29`, `i22` |
| **sensor_id** | `rs1`, `rs2`, … — node within a scene; **not** a detection target |
| **Profile** | 14-dim mean conf + accept rate fingerprint per **scene** |
| **Threshold bank** | Optimized \(H_i\) per **scene** |
| **Hysteresis** | Require K consecutive scene wins before swapping thresholds |
| **Oracle** | Use true `scene_id` (upper bound) |
| **Accept rate** | Fraction of Ki inferences with conf ≥ \(H_i\) (not IDK) |
