"""Runtime inference helpers: Ki forward + Kdet fallback (see cascade/kdet.py)."""

from __future__ import annotations

import torch
from torch import nn

from cascade.kdet import (
    KDET_WCET_MS,
    KDET_MODES,
    KdetContext,
    load_kdet_context_from_metrics,
    oracle_context,
    run_kdet,
    run_kdet_stub,
)
from utils.labels import is_deterministic_ki, threshold_hi_for_ki


@torch.inference_mode()
def ki_forward_outcome(
    model: nn.Module,
    mic: torch.Tensor,
    geo: torch.Tensor | None,
    modality: str,
    class_names: list[str],
    threshold_hi: float,
    device: torch.device,
) -> str:
    """
    Run one Ki on one sample.

    Returns "IDK" if max(softmax) < H_i, else the predicted label string.
    """
    model.eval()
    mic_b = mic.unsqueeze(0).to(device, non_blocking=True)

    if modality == "mic":
        logits = model(mic_b)
    else:
        assert geo is not None
        geo_b = geo.unsqueeze(0).to(device, non_blocking=True)
        logits = model(mic_b, geo_b)

    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)

    if float(conf.item()) < threshold_hi:
        return "IDK"

    return class_names[int(pred_idx.item())]


def threshold_for_record(
    ki_name: str,
    registry_threshold: float | None,
) -> float | None:
    """Registry H_i wins; fall back to paper defaults."""
    if is_deterministic_ki(ki_name):
        return None
    if registry_threshold is not None:
        return registry_threshold
    return threshold_hi_for_ki(ki_name)
