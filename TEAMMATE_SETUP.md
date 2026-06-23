# IDK Cascade Classifiers — Teammate Handoff

Trained **Ki** classifiers (K0–K6) plus deterministic fallback **Kdet** for the hierarchical IDK cascade.

## What is in this package

| File | Purpose |
|------|---------|
| `K0.pt` … `K6.pt` | Trained neural network weights per classifier |
| `Kdet.pt` | Deterministic fallback (no IDK deferral) |
| `classifier_registry.json` | Load paths, class names, H_i thresholds, routing DAG, timing |
| `K*_metrics.json`, `Kdet_metrics.json` | Validation accuracy, F1, confusion matrices |
| `wcet_profile.json` | Worst-case execution time (WCET) profile per Ki |

## Quick start

1. Clone the code repo: https://github.com/Borgatlat/IDKCascades
2. Unzip this bundle into the repo root (merge the `checkpoints/` folder).
3. Install dependencies (`torch`, etc.) matching the project environment.
4. Load models:

```python
from pathlib import Path
from cascade.loader import load_cascade_models

models, registry, device = load_cascade_models(
    checkpoint_dir=Path("checkpoints"),
    registry_path=Path("checkpoints/classifier_registry.json"),
)
# models["K0"], models["K1"], ... models["Kdet"]
```

## Classifier summary

| Ki | Level | Classes |
|----|-------|---------|
| K0, K1 | Intermediate | SUV, Coupe, Background |
| K2, K3 | Global | GLE350, CX-30, Mustang, MX-5, Background |
| K4 | Specialized SUV | GLE350, CX-30 |
| K5, K6 | Specialized Coupe | Mustang, MX-5 |
| Kdet | Deterministic | All base classes (mic + vision) |

Research figures (confusion matrices, registry table) are in `checkpoints/figures/` on GitHub.

## Notes

- Preprocessed **dataset** is not included (too large). Ask for dataset access separately if you need retraining.
- Registry checkpoint paths use `checkpoints\Ki.pt` — keep files in `checkpoints/` relative to repo root.
