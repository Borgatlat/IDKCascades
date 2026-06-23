# IDK Cascade Classifiers — Teammate Handoff

Trained **Ki** classifiers (K0–K6) plus deterministic fallback **Kdet** for the hierarchical IDK cascade.

## What is in this package

| File | Purpose |
|------|---------|
| `checkpoints/weights/K0_weights.pt` … `K6_weights.pt`, `Kdet_weights.pt` | **Recommended:** flat model weights (state_dict only) |
| `checkpoints/weights/all_weights.pt` | Single file with all 8 classifiers `{Ki: state_dict}` |
| `checkpoints/weights/manifest.json` | Param counts, class names, SHA256 checksums |
| `checkpoints/K0.pt` … `Kdet.pt` | Full training checkpoints (weights inside `model_state_dict`) |
| `classifier_registry.json` | Load paths, class names, H_i thresholds, routing DAG, timing |
| `K*_metrics.json`, `Kdet_metrics.json` | Validation accuracy, F1, confusion matrices |
| `wcet_profile.json` | Worst-case execution time (WCET) profile per Ki |

## Important: how to load weights

Full checkpoints (`K0.pt`) are **not** a raw state_dict. They are a Python dict:

```python
{
  "ki": "K0",
  "model_state_dict": { ... actual tensors ... },
  "class_names": [...],
  ...
}
```

**Wrong (common mistake):**
```python
model.load_state_dict(torch.load("checkpoints/K0.pt"))  # fails — wrong keys
```

**Correct — use flat weight files:**
```python
import torch
from models.dual_modal_cnn import build_ki_model

model = build_ki_model("K0", num_classes=3)
weights = torch.load("checkpoints/weights/K0_weights.pt", map_location="cpu")
model.load_state_dict(weights)
model.eval()
```

**Correct — use project loader (handles both formats):**
```python
from pathlib import Path
from cascade.loader import load_cascade_models

models, registry, device = load_cascade_models(
    checkpoint_dir=Path("checkpoints"),
    registry_path=Path("checkpoints/classifier_registry.json"),
)
```

**Correct — load all weights from one bundle:**
```python
bundle = torch.load("checkpoints/weights/all_weights.pt", map_location="cpu")
model = build_ki_model("K3", num_classes=5)
model.load_state_dict(bundle["K3"])
```

## Verify weights on your machine

```bash
python verify_classifiers.py
```

You should see `[OK]` for K0–K6 and Kdet.

## Quick start

1. Clone the code repo: https://github.com/Borgatlat/IDKCascades
2. Unzip this bundle into the repo root (merge the `checkpoints/` folder).
3. Install dependencies (`torch`, etc.) matching the project environment.
4. Run `python verify_classifiers.py` to confirm weights load.

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
- Keep `checkpoints/` and `checkpoints/weights/` relative to the repo root.
