"""Step 2: validate EXPAND output vs old marginal/cheap-Kdet baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.classifier_registry import KDET

# Old setup before teammate fix (real Kdet model, marginal stats only).
OLD_KDET_WCET_MS = 144.39
OLD_KDET_RUNTIME_MS = 28.28
NEW_KDET_WCET_MS = 10_000.0


def cascade_ends_with_kdet(sequence: list[str]) -> bool:
    return len(sequence) > 0 and sequence[-1] == KDET


def kdet_not_first_unless_only_step(sequence: list[str]) -> bool:
    """Kdet may be the sole step for degenerate branches (e.g. K1:background)."""
    if len(sequence) <= 1:
        return True
    return sequence[0] != KDET


def validate_probability_tables(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    initial = raw["initial"]
    specialized = raw.get("specialized", {})
    return {
        "path": str(path.resolve()),
        "initial_columns": initial.get("columns", []),
        "initial_row_count": len(initial.get("rows", [])),
        "initial_total_samples": initial.get("total_samples", 0),
        "specialized_branches": {
            branch: {
                "columns": table.get("columns", []),
                "row_count": len(table.get("rows", [])),
                "total_samples": table.get("total_samples", 0),
            }
            for branch, table in specialized.items()
        },
        "wcet_ms_keys": sorted(raw.get("wcet_ms", {}).keys()),
        "kdet_config": raw.get("kdet", {}),
        "checks": {
            "has_all_initial_columns": set(initial.get("columns", [])) >= {"K0", "K1", "K2", "K3"},
            "initial_rows_non_empty": len(initial.get("rows", [])) > 0,
            "sample_count_at_least_1000": initial.get("total_samples", 0) >= 1000,
        },
    }


def validate_cascades(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    initial = raw["initial_cascade"]
    specialized = raw["specialized_cascades"]

    all_sequences = {"initial": initial, **specialized}
    per_chain = {}
    for name, seq in all_sequences.items():
        per_chain[name] = {
            "sequence": seq,
            "length": len(seq),
            "ends_with_kdet": cascade_ends_with_kdet(seq),
            "kdet_not_first": kdet_not_first_unless_only_step(seq),
            "kdet_count": seq.count(KDET),
        }

    return {
        "path": str(path.resolve()),
        "expand_cost_ms": raw.get("expand_cost_ms"),
        "initial_cascade": initial,
        "specialized_cascades": specialized,
        "per_chain": per_chain,
        "checks": {
            "all_end_with_kdet": all(v["ends_with_kdet"] for v in per_chain.values()),
            "no_kdet_first": all(v["kdet_not_first"] for v in per_chain.values()),
            "initial_kdet_last_only": initial[-1] == KDET and initial[0] != KDET,
            "specialized_use_hierarchy": any(
                len(seq) > 2 for k, seq in specialized.items() if k != "K1:background"
            ),
        },
    }


def compare_old_vs_new() -> dict:
    """Explain why old Athan-style setup over-used Kdet."""
    return {
        "old_kdet": {
            "wcet_ms": OLD_KDET_WCET_MS,
            "runtime_ms": OLD_KDET_RUNTIME_MS,
            "p_correct": 0.9402,
            "optimizer_input": "marginal p_idk / p_correct per Ki (classifier_registry.json)",
            "problem": "Kdet looked cheap (~28–144 ms) and reliable → picked too early",
        },
        "new_kdet": {
            "wcet_ms": NEW_KDET_WCET_MS,
            "p_correct": 1.0,
            "optimizer_input": "joint probability_tables.json (cross-ref on same samples)",
            "fix": "Kdet is terminal penalty (10 s); EXPAND builds Ki chain first",
        },
        "wcet_ratio_kdet_vs_k2": round(NEW_KDET_WCET_MS / 8.74, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EXPAND optimizer outputs")
    parser.add_argument("--tables", type=Path, default=Path("checkpoints/probability_tables.json"))
    parser.add_argument("--cascades", type=Path, default=Path("checkpoints/synthesized_cascades.json"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/optimizer_validation.json"))
    args = parser.parse_args()

    tables_report = validate_probability_tables(args.tables)
    cascades_report = validate_cascades(args.cascades)
    comparison = compare_old_vs_new()

    all_checks = {
        **{f"tables_{k}": v for k, v in tables_report["checks"].items()},
        **{f"cascades_{k}": v for k, v in cascades_report["checks"].items()},
    }
    passed = all(all_checks.values())

    report = {
        "validation_passed": passed,
        "checks": all_checks,
        "probability_tables": tables_report,
        "synthesized_cascades": cascades_report,
        "old_vs_new": comparison,
        "summary": {
            "val_samples_profiled": tables_report["initial_total_samples"],
            "joint_outcome_rows": tables_report["initial_row_count"],
            "expand_expected_initial_cost_ms": cascades_report["expand_cost_ms"],
            "initial_cascade": cascades_report["initial_cascade"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation {'PASSED' if passed else 'FAILED'}")
    print(f"  Profiled samples: {tables_report['initial_total_samples']}")
    print(f"  Joint outcome rows (initial): {tables_report['initial_row_count']}")
    print(f"  EXPAND expected cost: {cascades_report['expand_cost_ms']:.2f} ms")
    print(f"  Initial cascade: {' -> '.join(cascades_report['initial_cascade'])}")
    print(f"  All chains end with Kdet: {cascades_report['checks']['all_end_with_kdet']}")
    print(f"  Kdet never first: {cascades_report['checks']['no_kdet_first']}")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
