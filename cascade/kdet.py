"""Kdet fallback: oracle (profiling), registry (simulated ~94%), or live model.

Paper usage:
  - oracle:  EXPAND probability-table profiling only (always correct, P=1.0).
  - registry: default runtime eval — deterministic sample from validation CM (~94%).
  - model:   real Kdet.pt forward pass (best for final numbers on target hardware).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.labels import GLOBAL_CLASS_NAMES, KI_REGISTRY

# Paper penalty WCET for Kdet (ms). Override via wcet_profile.json at runtime.
KDET_WCET_MS = 10_000.0

# oracle = perfect labels (probability-table profiling only)
# registry = sample from validation confusion matrix (~94% acc, reproducible per sample)
# model = forward pass through trained Kdet weights
KDET_MODES = ("oracle", "registry", "model")


@dataclass
class KdetContext:
    """How Kdet resolves when the cascade reaches the deterministic fallback."""

    mode: str = "registry"
    sleep: bool = False
    wcet_ms: float = KDET_WCET_MS
    class_names: list[str] = field(default_factory=lambda: list(GLOBAL_CLASS_NAMES))
    # registry mode: true_label -> normalized row of P(pred | true)
    row_probs: dict[str, np.ndarray] | None = None
    model: nn.Module | None = None
    modality: str = "both"
    val_accuracy: float | None = None  # from Kdet_metrics.json best_metrics

    def describe(self) -> str:
        if self.mode == "oracle":
            return "oracle stub (P=1.0, profiling only — not for runtime eval)"
        if self.mode == "registry":
            acc = f"{self.val_accuracy * 100:.1f}%" if self.val_accuracy is not None else "~94%"
            return f"simulated Kdet via validation CM ({acc} val accuracy, reproducible per sample)"
        return "real Kdet model forward pass"


def load_kdet_context_from_metrics(
    metrics_path: Path,
    *,
    mode: str = "registry",
    sleep: bool = False,
    wcet_ms: float = KDET_WCET_MS,
) -> KdetContext:
    """Build registry-mode context from Kdet_metrics.json validation confusion matrix."""
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = raw["best_metrics"]
    class_names = list(best["present_classes"])
    cm = np.array(best["confusion_matrix"], dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    probs = np.divide(cm, row_sums, where=row_sums > 0)

    row_probs: dict[str, np.ndarray] = {}
    for i, name in enumerate(class_names):
        row_probs[name] = probs[i]

    val_acc = float(best.get("accuracy", 0.0))

    return KdetContext(
        mode=mode,
        sleep=sleep,
        wcet_ms=wcet_ms,
        class_names=class_names,
        row_probs=row_probs,
        modality=KI_REGISTRY["Kdet"].modality,
        val_accuracy=val_acc,
    )


def oracle_context(*, sleep: bool = False, wcet_ms: float = KDET_WCET_MS) -> KdetContext:
    return KdetContext(mode="oracle", sleep=sleep, wcet_ms=wcet_ms)


@torch.inference_mode()
def _kdet_model_forward(
    ctx: KdetContext,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor | None,
    device: torch.device,
) -> tuple[str, float]:
    """Run trained Kdet; always commits (no IDK)."""
    assert ctx.model is not None
    ctx.model.eval()
    mic_b = mic_t.unsqueeze(0).to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if ctx.modality == "mic":
        logits = ctx.model(mic_b)
    else:
        assert geo_t is not None
        geo_b = geo_t.unsqueeze(0).to(device, non_blocking=True)
        logits = ctx.model(mic_b, geo_b)

    if device.type == "cuda":
        torch.cuda.synchronize()
    measured_ms = (time.perf_counter() - t0) * 1000.0

    pred_idx = int(logits.argmax(dim=1).item())
    return ctx.class_names[pred_idx], measured_ms


def _kdet_registry_sample(ctx: KdetContext, true_label: str, sample_key: int) -> str:
    """Deterministic draw from validation CM row for true_label."""
    assert ctx.row_probs is not None
    if true_label not in ctx.row_probs:
        return true_label
    weights = ctx.row_probs[true_label]
    rng = random.Random(int(sample_key))
    idx = rng.choices(range(len(ctx.class_names)), weights=weights.tolist(), k=1)[0]
    return ctx.class_names[idx]


def run_kdet(
    ctx: KdetContext,
    true_label: str,
    *,
    sample_key: int = 0,
    mic_t: torch.Tensor | None = None,
    geo_t: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[str, float]:
    """
    Invoke Kdet fallback.

    Returns (predicted_global_label, latency_ms).
    """
    if ctx.mode == "model":
        if ctx.model is None or mic_t is None or device is None:
            raise ValueError("Kdet model mode requires model, mic_t, and device")
        pred, measured = _kdet_model_forward(ctx, mic_t, geo_t, device)
        if ctx.sleep:
            time.sleep(ctx.wcet_ms / 1000.0)
            return pred, ctx.wcet_ms
        return pred, measured

    if ctx.sleep:
        time.sleep(ctx.wcet_ms / 1000.0)

    if ctx.mode == "oracle":
        return true_label, ctx.wcet_ms

    if ctx.mode == "registry":
        return _kdet_registry_sample(ctx, true_label, sample_key), ctx.wcet_ms

    raise ValueError(f"Unknown Kdet mode: {ctx.mode!r}. Use one of {KDET_MODES}")


def build_kdet_context(
    mode: str,
    *,
    metrics_path: Path = Path("checkpoints/Kdet_metrics.json"),
    checkpoint_dir: Path = Path("checkpoints"),
    registry_path: Path | None = None,
    sleep: bool = False,
    wcet_ms: float | None = None,
) -> KdetContext:
    """Construct KdetContext for cascade evaluation."""
    from cascade.loader import load_kdet_model

    wcet = wcet_ms if wcet_ms is not None else KDET_WCET_MS
    if mode == "oracle":
        return oracle_context(sleep=sleep, wcet_ms=wcet)
    if mode == "registry":
        return load_kdet_context_from_metrics(metrics_path, mode="registry", sleep=sleep, wcet_ms=wcet)
    if mode == "model":
        ctx = load_kdet_context_from_metrics(metrics_path, mode="model", sleep=sleep, wcet_ms=wcet)
        model, _ = load_kdet_model(checkpoint_dir, registry_path)
        return KdetContext(
            mode="model",
            sleep=sleep,
            wcet_ms=wcet,
            class_names=ctx.class_names,
            model=model,
            modality=ctx.modality,
        )
    raise ValueError(f"Unknown kdet mode {mode!r}; use {KDET_MODES}")


def run_kdet_stub(ground_truth_label: str, *, sleep: bool = True) -> tuple[str, float]:
    """Oracle Kdet for probability-table profiling only (always correct)."""
    return run_kdet(oracle_context(sleep=sleep), ground_truth_label)
