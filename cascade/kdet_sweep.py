"""Sweep Kdet cost and measure none/RF/MLP performance at each value, with
optimizer assumption, skip-label cost, and runtime cost all held consistent
within each sweep point (this is the discipline that matters: a cascade
built assuming Kdet=1000ms must be trained and benchmarked against that same
1000ms, not a mix of assumptions).

For each Kdet cost K in the sweep list:
  1. Rebuild the cascade + skip datasets assuming Kdet costs K (cost-aware
     labels by default, since correctness-only labels already showed they
     can lose to baseline -- see conversation history).
  2. Retrain RF and MLP skippers against that cascade shape.
  3. Benchmark none/rf/mlp with the runtime Kdet override also set to K, so
     the "Kdet" the runtime actually pays matches what the optimizer/skipper
     assumed when built.

Usage
-----
    python -m cascade.kdet_sweep
    python -m cascade.kdet_sweep --costs 10.85 100 1000 10000
    python -m cascade.kdet_sweep --costs 10.85 1000 --max-samples 1000
"""

from __future__ import annotations

from pathlib import Path

from cascade.runtime_cascade import benchmark
from cascade.skipper_dataset import DATASET_DIR, build_all_position_datasets
from cascade.train_skippers import (
    MLP_SKIPPER_DIR,
    RF_SKIPPER_DIR,
    train_all_mlp_skippers,
    train_all_rf_skippers,
)

DEFAULT_SWEEP_COSTS = [10.85, 100.0, 1000.0, 10000.0]


def run_sweep(
    costs: list[float] = DEFAULT_SWEEP_COSTS,
    cost_aware: bool = True,
    max_samples: int | None = None,
) -> list[dict]:
    results = []

    for k in costs:
        print(f"\n{'=' * 70}")
        print(f"KDET COST = {k} ms  (optimizer + dataset labels + runtime, all consistent)")
        print(f"{'=' * 70}")

        print(f"\n--- [1/3] rebuilding cascade + skip datasets (cost_aware={cost_aware}) ---")
        build_all_position_datasets(
            detector_mode="paper",
            detector_cost_ms=k,
            cost_aware=cost_aware,
        )

        print(f"\n--- [2/3] retraining skippers ---")
        train_all_rf_skippers(DATASET_DIR, RF_SKIPPER_DIR)
        train_all_mlp_skippers(DATASET_DIR, MLP_SKIPPER_DIR)

        print(f"\n--- [3/3] benchmarking ---")
        for mode in ["none", "rf", "mlp"]:
            print(f"\n>>> kdet={k}ms  mode={mode}")
            summary = benchmark(
                skip_mode=mode,
                detector_mode="paper",
                detector_cost_ms=k,
                detector_runtime_override_ms=k,
                max_samples=max_samples,
            )
            results.append({
                "kdet_cost_ms": k,
                "skip_mode": mode,
                "accuracy": summary["accuracy"],
                "avg_runtime_ms": summary["average_runtime_ms"],
                "total": summary["total"],
            })

    print(f"\n\n{'=' * 70}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Kdet cost (ms)':>15} {'mode':>6} {'accuracy':>10} {'avg runtime (ms)':>18}")
    for r in results:
        print(f"{r['kdet_cost_ms']:>15.2f} {r['skip_mode']:>6} {r['accuracy']:>10.4f} {r['avg_runtime_ms']:>18.3f}")

    # Per-Kdet-cost: did RF/MLP beat baseline?
    print(f"\n{'-' * 70}")
    print("DID SKIPPING WIN AT EACH KDET COST?")
    print(f"{'-' * 70}")
    by_cost: dict[float, dict[str, dict]] = {}
    for r in results:
        by_cost.setdefault(r["kdet_cost_ms"], {})[r["skip_mode"]] = r
    for k, modes in by_cost.items():
        baseline = modes.get("none", {}).get("avg_runtime_ms")
        if baseline is None:
            continue
        for mode in ["rf", "mlp"]:
            if mode not in modes:
                continue
            delta = baseline - modes[mode]["avg_runtime_ms"]
            verdict = "WIN" if delta > 0 else "LOSE"
            print(f"  kdet={k:>10.2f}ms  {mode:>3}: baseline={baseline:.3f}ms  "
                  f"{mode}={modes[mode]['avg_runtime_ms']:.3f}ms  "
                  f"delta={delta:+.3f}ms  [{verdict}]")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", type=float, nargs="+", default=DEFAULT_SWEEP_COSTS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cost-aware", action="store_true",
                        help="Use plain first-acceptor labels instead of cost-aware ones")
    args = parser.parse_args()

    run_sweep(
        costs=args.costs,
        cost_aware=not args.no_cost_aware,
        max_samples=args.max_samples,
    )
