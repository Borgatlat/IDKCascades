"""EXPAND cascade synthesis (paper Section IV, Algorithm 1–2)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from utils.classifier_registry import KDET
from utils.probability_tables import (
    ProbabilityTableBundle,
    cond_pr,
    cond_pr_specialized,
    prefix_all_idk,
    pr_idk_given_prefix,
)

# Paper partition for this project (must match profile_probability_tables.py).
KI_INTERMEDIATE = frozenset({"K0", "K1"})
KPHI_GLOBAL = frozenset({"K2", "K3"})
INITIAL_POOL = KI_INTERMEDIATE | KPHI_GLOBAL

INTERMEDIATE_CLASSES = ("suv", "coupe", "background")
GLOBAL_BASE_OUTCOMES = ("gle350", "cx30", "mustang", "miata", "background")

SPECIALIZED_BY_BRANCH: dict[str, frozenset[str]] = {
    "suv": frozenset({"K4"}),
    "coupe": frozenset({"K5", "K6"}),
    "background": frozenset(),
}


@dataclass
class CascadePlan:
    """Ordered Ki list ending with Kdet."""

    name: str
    sequence: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """Full EXPAND output: initial + specialized cascades."""

    initial: CascadePlan
    specialized: dict[str, CascadePlan] = field(default_factory=dict)
    expand_cost_ms: float = 0.0


def load_optimizer_inputs(tables_path: Path) -> ProbabilityTableBundle:
    return ProbabilityTableBundle.load(tables_path)


def _wcet(bundle: ProbabilityTableBundle, ki: str) -> float:
    return float(bundle.wcet_ms.get(ki, bundle.kdet.get("wcet_ms", 10_000.0)))


def _initial_rows(bundle: ProbabilityTableBundle) -> list[dict]:
    return bundle.initial.get("rows", [])


def _specialized_rows(bundle: ProbabilityTableBundle, branch: str) -> list[dict]:
    branch_table = bundle.specialized.get(branch, {})
    return branch_table.get("rows", [])


class ExpandOptimizer:
    """Memoized EXPAND / EXPAND′ over a ProbabilityTableBundle."""

    def __init__(self, bundle: ProbabilityTableBundle) -> None:
        self.bundle = bundle
        self.initial_rows = _initial_rows(bundle)
        self.c_det = _wcet(bundle, KDET)
        self._expand_cache: dict[frozenset[str], tuple[float, str]] = {}
        self._prime_cache: dict[tuple, tuple[float, str]] = {}

    def expand(self, s: frozenset[str]) -> tuple[float, str]:
        if s in self._expand_cache:
            return self._expand_cache[s]
        result = self._expand_compute(s)
        self._expand_cache[s] = result
        return result

    def _expand_compute(self, s: frozenset[str]) -> tuple[float, str]:
        if s == INITIAL_POOL:
            return self.c_det, KDET

        prefix = prefix_all_idk(s)
        best_cost = float("inf")
        best_next = KDET

        # Intermediate classifiers KI \ S (Algorithm 1, lines 8–12).
        for kj in sorted(KI_INTERMEDIATE - s):
            if kj not in self.bundle.initial.get("columns", []):
                continue
            tmp = self._step_cost_initial_intermediate(s, prefix, kj)
            if tmp < best_cost:
                best_cost = tmp
                best_next = kj

        # Global classifiers Kphi \ S (Algorithm 1, lines 13–17).
        for kj in sorted(KPHI_GLOBAL - s):
            if kj not in self.bundle.initial.get("columns", []):
                continue
            tmp = self._step_cost_initial_global(s, prefix, kj)
            if tmp < best_cost:
                best_cost = tmp
                best_next = kj

        if self.c_det < best_cost:
            best_cost = self.c_det
            best_next = KDET

        return best_cost, best_next

    def _step_cost_initial_intermediate(
        self,
        s: frozenset[str],
        prefix: dict[str, str],
        kj: str,
    ) -> float:
        cj = _wcet(self.bundle, kj)
        pr_idk = pr_idk_given_prefix(self.initial_rows, prefix, kj)
        cost_idk, _ = self.expand(frozenset(set(s) | {kj}))

        cost = cj + pr_idk * cost_idk
        for il in INTERMEDIATE_CLASSES:
            pr_il = cond_pr(self.initial_rows, prefix, kj, il)
            if pr_il <= 0.0:
                continue
            cost_il, _ = self.expand_prime(s, il, frozenset(), kj)
            cost += pr_il * cost_il
        return cost

    def _step_cost_initial_global(
        self,
        s: frozenset[str],
        prefix: dict[str, str],
        kj: str,
    ) -> float:
        cj = _wcet(self.bundle, kj)
        pr_idk = pr_idk_given_prefix(self.initial_rows, prefix, kj)
        cost_idk, _ = self.expand(frozenset(set(s) | {kj}))
        # Base-class outcomes terminate the initial phase (zero remaining cost).
        return cj + pr_idk * cost_idk

    def expand_prime(
        self,
        s: frozenset[str],
        intermediate: str,
        t: frozenset[str],
        kh: str,
    ) -> tuple[float, str]:
        key = (s, intermediate, t, kh)
        if key in self._prime_cache:
            return self._prime_cache[key]
        result = self._expand_prime_compute(s, intermediate, t, kh)
        self._prime_cache[key] = result
        return result

    def _expand_prime_compute(
        self,
        s: frozenset[str],
        intermediate: str,
        t: frozenset[str],
        kh: str,
    ) -> tuple[float, str]:
        k_l = SPECIALIZED_BY_BRANCH.get(intermediate, frozenset())
        universe = k_l | KPHI_GLOBAL
        if (s | t) >= universe or not _specialized_rows(self.bundle, intermediate):
            return self.c_det, KDET

        prefix = prefix_all_idk(s | t)
        rows = _specialized_rows(self.bundle, intermediate)
        best_cost = float("inf")
        best_next = KDET

        candidates = sorted((k_l - t) | (KPHI_GLOBAL - s))
        for kj in candidates:
            if kj not in self.bundle.specialized.get(intermediate, {}).get("columns", []):
                continue
            tmp = self._step_cost_specialized(
                s, intermediate, t, kh, prefix, rows, kj, k_l
            )
            if tmp < best_cost:
                best_cost = tmp
                best_next = kj

        if self.c_det < best_cost:
            best_cost = self.c_det
            best_next = KDET

        return best_cost, best_next

    def _step_cost_specialized(
        self,
        s: frozenset[str],
        intermediate: str,
        t: frozenset[str],
        kh: str,
        prefix: dict[str, str],
        rows: list[dict],
        kj: str,
        k_l: frozenset[str],
    ) -> float:
        cj = _wcet(self.bundle, kj)
        pr_idk = cond_pr_specialized(rows, prefix, kh, intermediate, kj, "IDK")

        if kj in k_l:
            cost_idk, _ = self.expand_prime(s, intermediate, frozenset(set(t) | {kj}), kh)
        else:
            cost_idk, _ = self.expand_prime(frozenset(set(s) | {kj}), intermediate, t, kh)

        cost = cj + pr_idk * cost_idk

        # Base-class termination: remaining cost is zero.
        for base in GLOBAL_BASE_OUTCOMES:
            pr_base = cond_pr_specialized(rows, prefix, kh, intermediate, kj, base)
            cost += pr_base * 0.0

        return cost


def synthesize_initial_cascade(bundle: ProbabilityTableBundle) -> CascadePlan:
    """Algorithm 2 lines 3–9."""
    opt = ExpandOptimizer(bundle)
    s: set[str] = set()
    sequence: list[str] = []

    while True:
        _, next_ki = opt.expand(frozenset(s))
        sequence.append(next_ki)
        if next_ki == KDET:
            break
        s.add(next_ki)

    return CascadePlan(name="initial", sequence=sequence)


def synthesize_specialized_cascade(
    bundle: ProbabilityTableBundle,
    upstream_ki: str,
    intermediate_class: str,
    initial_cascade: list[str],
) -> CascadePlan:
    """Algorithm 2 lines 10–21."""
    if upstream_ki not in KI_INTERMEDIATE:
        raise ValueError(f"{upstream_ki} is not an intermediate classifier")

    if upstream_ki not in initial_cascade:
        raise ValueError(f"{upstream_ki} not in initial cascade {initial_cascade}")

    opt = ExpandOptimizer(bundle)
    idx = initial_cascade.index(upstream_ki)
    s = frozenset(initial_cascade[:idx])
    t: set[str] = set()
    sequence: list[str] = []

    while True:
        _, next_ki = opt.expand_prime(s, intermediate_class, frozenset(t), upstream_ki)
        sequence.append(next_ki)
        if next_ki == KDET:
            break
        if next_ki in KPHI_GLOBAL:
            s = frozenset(set(s) | {next_ki})
        else:
            t.add(next_ki)

    name = f"specialized_{upstream_ki}_{intermediate_class}"
    return CascadePlan(name=name, sequence=sequence)


def synthesize_all(bundle: ProbabilityTableBundle) -> SynthesisResult:
    """Run full EXPAND preprocessing from the paper."""
    opt = ExpandOptimizer(bundle)
    initial_cost, _ = opt.expand(frozenset())
    initial = synthesize_initial_cascade(bundle)

    specialized: dict[str, CascadePlan] = {}
    for ki in KI_INTERMEDIATE:
        if ki not in initial.sequence:
            continue
        for il in INTERMEDIATE_CLASSES:
            if il not in bundle.specialized:
                continue
            plan = synthesize_specialized_cascade(bundle, ki, il, initial.sequence)
            specialized[f"{ki}:{il}"] = plan

    return SynthesisResult(
        initial=initial,
        specialized=specialized,
        expand_cost_ms=initial_cost,
    )


def expected_remaining_ms(
    bundle: ProbabilityTableBundle,
    prefix: dict[str, str],
    next_ki: str,
) -> float:
    """Expected remaining ms from empty S with one-step lookahead (debug helper)."""
    opt = ExpandOptimizer(bundle)
    s = frozenset(k for k, v in prefix.items() if v == "IDK")
    if next_ki in KI_INTERMEDIATE:
        return opt._step_cost_initial_intermediate(s, prefix, next_ki)
    return opt._step_cost_initial_global(s, prefix, next_ki)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize IDK cascades via EXPAND")
    parser.add_argument(
        "--tables",
        type=Path,
        default=Path("checkpoints/probability_tables.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/synthesized_cascades.json"),
    )
    args = parser.parse_args()

    tables_path = args.tables.resolve()
    if not tables_path.exists():
        raise FileNotFoundError(
            f"{tables_path} not found. Run: python profile_probability_tables.py"
        )

    bundle = load_optimizer_inputs(tables_path)
    if not bundle.initial.get("rows"):
        raise ValueError(
            "probability_tables.json has no initial rows. "
            "Run: python profile_probability_tables.py"
        )

    result = synthesize_all(bundle)
    payload = {
        "expand_cost_ms": result.expand_cost_ms,
        "initial_cascade": result.initial.sequence,
        "specialized_cascades": {
            key: plan.sequence for key, plan in result.specialized.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"EXPAND expected initial cost: {result.expand_cost_ms:.2f} ms")
    print(f"Initial cascade: {' -> '.join(result.initial.sequence)}")
    for key, plan in result.specialized.items():
        print(f"  {key}: {' -> '.join(plan.sequence)}")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
