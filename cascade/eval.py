"""Shared cascade evaluation helpers for run_cascade.py and plot_cascade_pareto.py."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch import nn

from cascade.executor import CascadePlan, CascadeTrace, ExecutorConfig, execute_sample, load_wcet_profile
from cascade.kdet import KdetContext
from cascade.loader import load_cascade_models
from profile_probability_tables import validation_indices
from training.trainer import get_device, load_spectrogram_cache


from utils.hardware import hardware_info


def sample_tensors(mic: np.ndarray, geo: np.ndarray, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match KiDataset layout: one sample -> (1, H, W) tensors."""
    mic_t = torch.from_numpy(mic[idx][None, :, :].copy())
    geo_t = torch.from_numpy(geo[idx][None, :, :].copy())
    return mic_t, geo_t


def evaluate_val_set(
    plan: CascadePlan,
    models: dict[str, nn.Module | None],
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata,
    val_idx: np.ndarray,
    device: torch.device,
    config: ExecutorConfig,
) -> list[CascadeTrace]:
    """Run synthesized cascade on each validation index."""
    traces: list[CascadeTrace] = []
    for idx in val_idx:
        row = metadata.iloc[int(idx)]
        true_global = str(row["global_label"])
        mic_t, geo_t = sample_tensors(mic, geo, int(idx))
        traces.append(
            execute_sample(
                plan, models, registry, mic_t, geo_t, device, true_global, config,
                sample_key=int(idx),
            )
        )
    return traces


def summarize_traces(traces: list[CascadeTrace], deadline_ms: float | None = None) -> dict:
    """Aggregate accuracy, latency, Kdet / override rates; optional deadline feasibility."""
    n = len(traces)
    if n == 0:
        return {"n": 0}

    latencies = [t.latency_ms for t in traces]
    ki_counter: Counter = Counter()
    for t in traces:
        ki_counter.update(t.fired_kis)

    summary: dict = {
        "n": n,
        "accuracy": sum(t.correct for t in traces) / n,
        "kdet_rate": sum(t.hit_kdet for t in traces) / n,
        "deadline_override_rate": sum(t.deadline_override for t in traces) / n,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        },
        "ki_fire_counts": dict(sorted(ki_counter.items())),
        "branch_counts": dict(Counter(t.branch_key for t in traces if t.branch_key)),
    }

    if deadline_ms is not None:
        feasible = sum(1 for t in traces if t.latency_ms <= deadline_ms)
        summary["deadline_ms"] = float(deadline_ms)
        summary["deadline_feasible_rate"] = feasible / n
        summary["deadline_miss_rate"] = 1.0 - summary["deadline_feasible_rate"]

    return summary


@dataclass
class EvalBundle:
    """Models + val split loaded once for repeated deadline sweeps."""

    plan: CascadePlan
    models: dict[str, nn.Module | None]
    registry: object
    mic: np.ndarray
    geo: np.ndarray
    metadata: object
    val_idx: np.ndarray
    device: torch.device
    wcet_ms: dict[str, float]


def load_eval_bundle(
    *,
    plan_path: Path,
    processed_dir: Path,
    checkpoint_dir: Path,
    registry_path: Path,
    seed: int = 42,
    max_samples: int | None = None,
) -> EvalBundle:
    """Load plan, models, and val split once for repeated deadline sweeps."""
    plan = CascadePlan.load(plan_path)
    wcet_ms = load_wcet_profile(checkpoint_dir)
    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)
    mic, geo, metadata = load_spectrogram_cache(processed_dir)
    val_idx = validation_indices(metadata, seed=seed)
    if max_samples is not None:
        val_idx = val_idx[:max_samples]

    return EvalBundle(
        plan=plan,
        models=models,
        registry=registry,
        mic=mic,
        geo=geo,
        metadata=metadata,
        val_idx=val_idx,
        device=device,
        wcet_ms=wcet_ms,
    )


def default_deadline_sweep(wcet_ms: dict[str, float], steps: int = 25) -> list[float]:
    """
    Build D_system grid spanning Ki-only region through Kdet WCET.

    Lower segment: cascade Ki latencies (meaningful guard / branch behavior).
    Upper segment: through Kdet WCET so feasible_rate can reach 1.0.
    """
    ki_wcets = [v for k, v in wcet_ms.items() if k != "Kdet" and v > 0]
    kdet_wcet = float(wcet_ms.get("Kdet", 10_000.0))
    if not ki_wcets:
        ki_wcets = [1.0]

    lo = min(ki_wcets)
    # Upper bound of Ki-only worst paths (exclude 10s Kdet tail from auto hi cap).
    ki_chain_hi = min(sum(sorted(ki_wcets, reverse=True)[:5]), 800.0)
    ki_chain_hi = max(ki_chain_hi, lo + 50.0)

    n_ki = max(steps // 2, 6)
    n_tail = max(steps - n_ki, 6)
    ki_points = np.linspace(lo, ki_chain_hi, n_ki)
    tail_points = np.linspace(ki_chain_hi, kdet_wcet * 1.05, n_tail)[1:]
    # Extra samples near Kdet WCET so guard plots resolve the transition cliff.
    near_kdet = np.linspace(0.85 * kdet_wcet, 1.05 * kdet_wcet, 8)

    merged = sorted({float(x) for x in np.concatenate([ki_points, tail_points, near_kdet])})
    return merged


def sweep_deadlines(
    bundle: EvalBundle,
    deadlines_ms: list[float],
    *,
    timing_mode: str = "table",
    deadline_guard: bool = True,
    kdet_sleep: bool = False,
    kdet: KdetContext | None = None,
) -> list[dict]:
    """
    Evaluate cascade at each D_system; returns one summary row per deadline.

    Guard off: single inference pass, feasible_rate computed post-hoc (fast).
    Guard on:  re-run per D (predictions / overrides can change).
    """
    if not deadline_guard:
        config = ExecutorConfig(
            timing_mode=timing_mode,
            wcet_ms=bundle.wcet_ms,
            deadline_ms=None,
            deadline_guard=False,
            kdet_sleep=kdet_sleep,
            kdet=kdet,
        )
        traces = evaluate_val_set(
            bundle.plan,
            bundle.models,
            bundle.registry,
            bundle.mic,
            bundle.geo,
            bundle.metadata,
            bundle.val_idx,
            bundle.device,
            config,
        )
        rows: list[dict] = []
        for i, deadline_ms in enumerate(deadlines_ms, start=1):
            summary = summarize_traces(traces, deadline_ms=deadline_ms)
            rows.append(summary)
            print(
                f"  [{i}/{len(deadlines_ms)}] D={deadline_ms:.1f} ms | "
                f"acc={summary['accuracy']:.3f} | "
                f"feasible={summary['deadline_feasible_rate']:.3f} | "
                f"override={summary['deadline_override_rate']:.3f} | "
                f"mean_lat={summary['latency_ms']['mean']:.1f} ms"
            )
        return rows

    rows = []
    for i, deadline_ms in enumerate(deadlines_ms, start=1):
        config = ExecutorConfig(
            timing_mode=timing_mode,
            wcet_ms=bundle.wcet_ms,
            deadline_ms=deadline_ms,
            deadline_guard=True,
            kdet_sleep=kdet_sleep,
            kdet=kdet,
        )
        traces = evaluate_val_set(
            bundle.plan,
            bundle.models,
            bundle.registry,
            bundle.mic,
            bundle.geo,
            bundle.metadata,
            bundle.val_idx,
            bundle.device,
            config,
        )
        summary = summarize_traces(traces, deadline_ms=deadline_ms)
        rows.append(summary)
        print(
            f"  [{i}/{len(deadlines_ms)}] D={deadline_ms:.1f} ms | "
            f"acc={summary['accuracy']:.3f} | "
            f"feasible={summary['deadline_feasible_rate']:.3f} | "
            f"override={summary['deadline_override_rate']:.3f} | "
            f"mean_lat={summary['latency_ms']['mean']:.1f} ms"
        )

    return rows
