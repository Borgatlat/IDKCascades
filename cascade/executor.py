"""Runtime cascade executor: walk synthesized_cascades.json on val samples."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from cascade.inference import KDET_WCET_MS, ki_forward_outcome, threshold_for_record
from cascade.kdet import KdetContext, run_kdet
from utils.labels import GLOBAL_CLASS_NAMES, INTERMEDIATE_CLASS_NAMES, KI_REGISTRY

# Ki that emit intermediate branch labels (suv / coupe / background).
INTERMEDIATE_KI = frozenset({"K0", "K1"})
INTERMEDIATE_LABELS = frozenset(INTERMEDIATE_CLASS_NAMES)


@dataclass(frozen=True)
class CascadePlan:
    """Ordered Ki lists produced by cascade/optimizer_stub.py (EXPAND)."""

    initial_cascade: list[str]
    specialized_cascades: dict[str, list[str]]
    expand_cost_ms: float = 0.0

    @classmethod
    def load(cls, path: Path) -> CascadePlan:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            initial_cascade=list(raw["initial_cascade"]),
            specialized_cascades={k: list(v) for k, v in raw["specialized_cascades"].items()},
            expand_cost_ms=float(raw.get("expand_cost_ms", 0.0)),
        )


@dataclass
class ExecutorConfig:
    """Controls timing source, Kdet sleep, and optional hard deadline guard."""

    timing_mode: str = "table"  # "table" | "live"
    wcet_ms: dict[str, float] = field(default_factory=dict)
    deadline_ms: float | None = None
    deadline_guard: bool = True
    kdet_sleep: bool = False
    kdet: KdetContext | None = None


@dataclass
class CascadeTrace:
    """Per-sample audit trail for accuracy + latency + schedulability analysis."""

    prediction: str
    true_label: str
    correct: bool
    fired_kis: list[str]
    hit_kdet: bool
    deadline_override: bool
    latency_ms: float
    branch_key: str | None = None


def load_wcet_profile(checkpoint_dir: Path) -> dict[str, float]:
    """Load per-Ki WCET from wcet_profile.json; Kdet uses paper stub cost."""
    wcet: dict[str, float] = {}
    path = checkpoint_dir / "wcet_profile.json"
    if path.exists():
        for entry in json.loads(path.read_text(encoding="utf-8")):
            wcet[entry["ki"]] = float(entry["wcet_ms"])
    # Kdet: use profiled WCET if present, else paper penalty stub.
    if "Kdet" not in wcet:
        wcet.setdefault("Kdet", KDET_WCET_MS)
    return wcet


def _wcet_remaining(sequence: list[str], start_idx: int, wcet_ms: dict[str, float]) -> float:
    """Sum WCET for Ki[start_idx:] — used by the schedulability guard."""
    total = 0.0
    for ki in sequence[start_idx:]:
        total += wcet_ms.get(ki, KDET_WCET_MS)
    return total


def _ki_cost(ki_name: str, measured_ms: float, config: ExecutorConfig) -> float:
    """Table mode: optimizer-aligned WCET; live mode: measured forward pass."""
    if config.timing_mode == "live":
        return measured_ms
    return config.wcet_ms.get(ki_name, KDET_WCET_MS)


@torch.inference_mode()
def _run_ki(
    ki_name: str,
    model: nn.Module,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor | None,
    registry,
    device: torch.device,
) -> tuple[str, float]:
    """
    Forward one Ki with RTS-compliant timing (CUDA sync + perf_counter).

    Returns (outcome_label_or_IDK, measured_latency_ms).
    """
    spec = KI_REGISTRY[ki_name]
    rec = registry.get(ki_name)
    hi = threshold_for_record(ki_name, rec.threshold_hi if rec else None)
    assert hi is not None, f"No threshold H_i for {ki_name}"

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outcome = ki_forward_outcome(
        model,
        mic_t,
        geo_t if spec.modality != "mic" else None,
        spec.modality,
        list(spec.class_names),
        hi,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    measured_ms = (time.perf_counter() - t0) * 1000.0
    return outcome, measured_ms


def _kdet_ctx(config: ExecutorConfig) -> KdetContext:
    if config.kdet is not None:
        return config.kdet
    from cascade.kdet import load_kdet_context_from_metrics

    return load_kdet_context_from_metrics(
        Path("checkpoints/Kdet_metrics.json"),
        sleep=config.kdet_sleep,
        wcet_ms=config.wcet_ms.get("Kdet", KDET_WCET_MS),
    )


def _invoke_kdet(
    config: ExecutorConfig,
    true_global: str,
    sample_key: int,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor,
    device: torch.device,
) -> tuple[str, float]:
    ctx = _kdet_ctx(config)
    return run_kdet(
        ctx,
        true_global,
        sample_key=sample_key,
        mic_t=mic_t,
        geo_t=geo_t,
        device=device,
    )


def _shortcut_kdet(
    true_global: str,
    fired: list[str],
    elapsed_ms: float,
    config: ExecutorConfig,
    sample_key: int,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor,
    device: torch.device,
) -> tuple[str, list[str], bool, bool, float]:
    """Deadline guard: skip remaining Ki and invoke Kdet immediately."""
    pred, kdet_ms = _invoke_kdet(config, true_global, sample_key, mic_t, geo_t, device)
    cost = _ki_cost("Kdet", kdet_ms, config)
    return pred, fired + ["Kdet"], True, True, elapsed_ms + cost


def _execute_sequence(
    sequence: list[str],
    models: dict[str, nn.Module | None],
    registry,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor,
    device: torch.device,
    true_global: str,
    specialized: dict[str, list[str]],
    config: ExecutorConfig,
    sample_key: int,
    *,
    fired: list[str] | None = None,
    elapsed_ms: float = 0.0,
    deadline_override: bool = False,
) -> tuple[str, list[str], bool, bool, float, str | None]:
    """
    Walk one ordered Ki list until a global prediction, branch, or Kdet.

    Returns:
      prediction, fired_kis, hit_kdet, deadline_override, latency_ms, branch_key
    """
    fired = list(fired or [])
    branch_key: str | None = None

    for i, ki_name in enumerate(sequence):
        # Hard real-time guard: D_remain must cover worst-case remainder of this sequence.
        if (
            config.deadline_ms is not None
            and config.deadline_guard
            and not deadline_override
        ):
            d_remain = config.deadline_ms - elapsed_ms
            wcet_rest = _wcet_remaining(sequence, i, config.wcet_ms)
            if d_remain < wcet_rest:
                pred, fired, hit_kdet, deadline_override, elapsed_ms = _shortcut_kdet(
                    true_global, fired, elapsed_ms, config, sample_key, mic_t, geo_t, device
                )
                return pred, fired, hit_kdet, deadline_override, elapsed_ms, branch_key

        if ki_name == "Kdet":
            pred, kdet_ms = _invoke_kdet(config, true_global, sample_key, mic_t, geo_t, device)
            elapsed_ms += _ki_cost("Kdet", kdet_ms, config)
            return pred, fired + ["Kdet"], True, deadline_override, elapsed_ms, branch_key

        model = models.get(ki_name)
        if model is None:
            raise ValueError(f"Missing weights for {ki_name}")

        outcome, measured_ms = _run_ki(ki_name, model, mic_t, geo_t, registry, device)
        fired.append(ki_name)
        elapsed_ms += _ki_cost(ki_name, measured_ms, config)

        if outcome == "IDK":
            continue

        # Intermediate Ki resolved a branch label → switch to specialized sub-cascade.
        if ki_name in INTERMEDIATE_KI and outcome in INTERMEDIATE_LABELS:
            branch_key = f"{ki_name}:{outcome}"
            sub = specialized.get(branch_key)
            if sub is None:
                raise KeyError(
                    f"No specialized cascade for {branch_key!r}. "
                    f"Re-run optimizer or check synthesized_cascades.json."
                )
            pred, fired, hit_kdet, deadline_override, elapsed_ms, _ = _execute_sequence(
                sub,
                models,
                registry,
                mic_t,
                geo_t,
                device,
                true_global,
                specialized,
                config,
                sample_key,
                fired=fired,
                elapsed_ms=elapsed_ms,
                deadline_override=deadline_override,
            )
            return pred, fired, hit_kdet, deadline_override, elapsed_ms, branch_key

        # Global / specialized Ki returned a concrete base-class label.
        if outcome in GLOBAL_CLASS_NAMES:
            return outcome, fired, False, deadline_override, elapsed_ms, branch_key

        raise ValueError(f"Unexpected outcome {outcome!r} from {ki_name}")

    # Sequence ended without Kdet (should not happen if plan ends with Kdet).
    pred, fired, hit_kdet, deadline_override, elapsed_ms = _shortcut_kdet(
        true_global, fired, elapsed_ms, config, sample_key, mic_t, geo_t, device
    )
    return pred, fired, hit_kdet, deadline_override, elapsed_ms, branch_key


@torch.inference_mode()
def execute_sample(
    plan: CascadePlan,
    models: dict[str, nn.Module | None],
    registry,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor,
    device: torch.device,
    true_global: str,
    config: ExecutorConfig,
    sample_key: int = 0,
) -> CascadeTrace:
    """Run the full initial → optional specialized → Kdet path for one sample."""
    pred, fired, hit_kdet, deadline_override, latency_ms, branch_key = _execute_sequence(
        plan.initial_cascade,
        models,
        registry,
        mic_t,
        geo_t,
        device,
        true_global,
        plan.specialized_cascades,
        config,
        sample_key,
    )
    return CascadeTrace(
        prediction=pred,
        true_label=true_global,
        correct=(pred == true_global),
        fired_kis=fired,
        hit_kdet=hit_kdet,
        deadline_override=deadline_override,
        latency_ms=latency_ms,
        branch_key=branch_key,
    )
