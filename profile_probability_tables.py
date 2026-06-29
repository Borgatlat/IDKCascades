"""Build joint probability tables from existing Ki weights (no retraining).

Run:
  python profile_probability_tables.py --max-samples 50   # smoke test
  python profile_probability_tables.py                    # full val set
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cascade.inference import ki_forward_outcome, threshold_for_record
from cascade.kdet import run_kdet_stub
from cascade.loader import load_cascade_models
from training.trainer import load_spectrogram_cache
from utils.labels import KI_REGISTRY
from utils.probability_tables import ProbabilityTableBundle, counter_to_rows, outcome_key
from utils.splits import apply_background_val_holdout, run_level_masks

# ---------------------------------------------------------------------------
# CUSTOMIZE: which Ki columns belong in each paper table.
# Think: "Which classifiers can appear in the initial cascade vs specialized branch?"
# ---------------------------------------------------------------------------
INITIAL_COLUMNS = ["K0", "K1", "K2", "K3"]

SPECIALIZED_COLUMNS: dict[str, list[str]] = {
    "suv": ["K4", "K2", "K3", "K1", "K0"],
    "coupe": ["K5", "K6", "K2", "K3", "K1", "K0"],
    "background": ["K2", "K3", "K1", "K0"],
}

IDK_KI_NAMES = [f"K{i}" for i in range(7)]  # K0..K6 — Kdet is simulated separately


def validation_indices(metadata, seed: int = 42) -> np.ndarray:
    """Same val split as training (K0 / all subset + background holdout)."""
    spec = KI_REGISTRY["K0"]
    train_mask, val_mask, _ = run_level_masks(metadata, spec=spec)
    train_mask, val_mask = apply_background_val_holdout(metadata, train_mask, val_mask, seed=seed)
    return np.where(val_mask)[0]


@torch.inference_mode()
def profile_sample_outcomes(
    models: dict[str, torch.nn.Module | None],
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    idx: int,
    device: torch.device,
) -> dict[str, str]:
    """Run every IDK classifier on one sample; return {Ki: outcome_string}."""
    # Match KiDataset: mic[:, None, :, :] -> shape (1, H, W) per sample
    mic_t = torch.from_numpy(mic[idx][None, :, :].copy())
    geo_t = torch.from_numpy(geo[idx][None, :, :].copy())
    outcomes: dict[str, str] = {}

    for ki_name in IDK_KI_NAMES:
        model = models[ki_name]
        if model is None:
            raise ValueError(f"Model {ki_name} is missing — check cascade/loader.py")

        rec = registry.get(ki_name)
        spec = KI_REGISTRY[ki_name]
        hi = threshold_for_record(ki_name, rec.threshold_hi if rec else None)
        assert hi is not None, f"No threshold for {ki_name}"

        outcomes[ki_name] = ki_forward_outcome(
            model,
            mic_t,
            geo_t if spec.modality != "mic" else None,
            spec.modality,
            list(spec.class_names),
            hi,
            device,
        )

    return outcomes


def load_timing_from_registry(checkpoint_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Pull measured C_i from wcet_profile.json on this machine."""
    wcet_ms: dict[str, float] = {}
    runtime_ms: dict[str, float] = {}
    wcet_path = checkpoint_dir / "wcet_profile.json"
    if wcet_path.exists():
        for entry in json.loads(wcet_path.read_text(encoding="utf-8")):
            wcet_ms[entry["ki"]] = float(entry["wcet_ms"])
            runtime_ms[entry["ki"]] = float(entry["avg_ms"])
    return wcet_ms, runtime_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile joint probability tables")
    parser.add_argument("--processed-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=Path("checkpoints/classifier_registry.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/probability_tables.json"))
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: cap val samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    registry_path = args.registry.resolve()

    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)
    mic, geo, metadata = load_spectrogram_cache(args.processed_dir)
    val_idx = validation_indices(metadata, seed=args.seed)
    if args.max_samples is not None:
        val_idx = val_idx[: args.max_samples]

    initial_counter: Counter = Counter()
    specialized_counters: dict[str, Counter] = {k: Counter() for k in SPECIALIZED_COLUMNS}
    branch_totals: dict[str, int] = {k: 0 for k in SPECIALIZED_COLUMNS}

    print(f"Profiling {len(val_idx)} validation samples on {device} ...")

    for n, idx in enumerate(val_idx, start=1):
        row = metadata.iloc[int(idx)]
        true_global = str(row["global_label"])
        intermediate = str(row["intermediate_label"])

        outcomes = profile_sample_outcomes(models, registry, mic, geo, int(idx), device)

        # Kdet: always correct, never IDK — no sleep during bulk profiling
        kdet_label, _ = run_kdet_stub(true_global, sleep=False)
        outcomes["Kdet"] = kdet_label

        initial_slice = {k: outcomes[k] for k in INITIAL_COLUMNS}
        initial_counter[outcome_key(initial_slice)] += 1

        if intermediate in specialized_counters:
            branch_totals[intermediate] += 1
            cols = SPECIALIZED_COLUMNS[intermediate]
            spec_slice = {k: outcomes[k] for k in cols}
            specialized_counters[intermediate][outcome_key(spec_slice)] += 1

        if n % 100 == 0:
            print(f"  {n}/{len(val_idx)}")

    total_initial = len(val_idx)
    wcet_ms, runtime_ms = load_timing_from_registry(checkpoint_dir)
    wcet_ms["Kdet"] = 10_000.0
    runtime_ms["Kdet"] = 10_000.0

    threshold_hi: dict[str, float] = {}
    for ki_name in IDK_KI_NAMES:
        rec = registry.get(ki_name)
        if rec and rec.threshold_hi is not None:
            threshold_hi[ki_name] = float(rec.threshold_hi)

    bundle = ProbabilityTableBundle(
        initial={
            "columns": INITIAL_COLUMNS,
            "total_samples": total_initial,
            "rows": counter_to_rows(initial_counter, total_initial),
        },
        specialized={
            branch: {
                "columns": SPECIALIZED_COLUMNS[branch],
                "total_samples": branch_totals[branch],
                "rows": counter_to_rows(
                    specialized_counters[branch],
                    max(branch_totals[branch], 1),
                ),
            }
            for branch in specialized_counters
        },
        wcet_ms=wcet_ms,
        runtime_ms=runtime_ms,
        threshold_hi=threshold_hi,
        kdet={
            "simulated": True,
            "wcet_ms": 10_000.0,
            "p_idk": 0.0,
            "p_correct": 1.0,
            "note": "Bulk profiling uses sleep=False; runtime cascade uses sleep=True",
        },
        meta={
            "processed_dir": str(args.processed_dir.resolve()),
            "checkpoint_dir": str(checkpoint_dir),
            "registry": str(registry_path),
            "device": str(device),
            "initial_columns": INITIAL_COLUMNS,
            "specialized_columns": SPECIALIZED_COLUMNS,
        },
    )

    bundle.save(args.out)
    print(f"Wrote cross-referenced tables -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
