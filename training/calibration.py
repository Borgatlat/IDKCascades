"""Precision calibration for IDK thresholds H_i (RTSS 2025 Section V-A).

Paper required precision: 0.95 (intermediate/specialized), 0.90 (global).
Calibration finds the lowest H_i on validation data such that
accuracy among samples with max(softmax) >= H_i meets that precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.classifier_registry import compute_p_idk


@torch.inference_mode()
def collect_val_confidences(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_confidence, is_correct) arrays over the validation loader."""
    model.eval()
    conf_list: list[float] = []
    correct_list: list[bool] = []

    for batch in loader:
        if modality == "mic":
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
        else:
            mic, geo, y = batch
            mic = mic.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(mic, geo)

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        conf_list.extend(conf.cpu().numpy().tolist())
        correct_list.extend((pred == y).cpu().numpy().tolist())

    return np.array(conf_list, dtype=np.float64), np.array(correct_list, dtype=bool)


def find_precision_calibrated_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    required_precision: float,
    *,
    min_answered: int = 10,
    tol: float = 1e-4,
    max_iter: int = 64,
) -> float:
    """
    Find the lowest H such that P(correct | conf >= H) >= required_precision.

    Lower H => more answers (higher coverage); raising H increases precision.
    We pick the minimum H that still satisfies the paper precision budget.
    """
    if len(confidences) == 0:
        return float(required_precision)

    lo, hi = 0.0, 1.0
    best = 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mask = confidences >= mid
        n = int(mask.sum())
        if n < min_answered:
            lo = mid
            continue
        precision = float(correct[mask].mean())
        if precision >= required_precision - tol:
            best = mid
            hi = mid
        else:
            lo = mid
    return float(best)


def calibrate_ki_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    required_precision: float,
    *,
    min_answered: int = 10,
) -> tuple[float, float, dict[str, Any]]:
    """
    Calibrate H_i on validation data; return (H_i, p_idk, diagnostics).

    p_idk is recomputed at the calibrated threshold via the registry helper.
    """
    confidences, correct = collect_val_confidences(model, loader, device, modality)
    hi = find_precision_calibrated_threshold(
        confidences,
        correct,
        required_precision,
        min_answered=min_answered,
    )
    p_idk = compute_p_idk(model, loader, device, modality, hi)
    answered = confidences >= hi
    precision = float(correct[answered].mean()) if answered.any() else 0.0
    return hi, p_idk, {
        "required_precision": required_precision,
        "achieved_precision": precision,
        "n_val": int(len(confidences)),
        "n_answered": int(answered.sum()),
    }
