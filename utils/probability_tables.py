"""Joint probability tables for paper EXPAND optimizer (Section III-B)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def outcome_key(outcomes: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """
    Turn {"K0": "IDK", "K1": "suv"} into a hashable tuple for counting.

    sorted() keeps the same combo from producing different keys.
    """
    return tuple(sorted(outcomes.items()))


def counter_to_rows(counter: Counter, total: int) -> list[dict[str, Any]]:
    """Convert raw counts into JSON-serializable rows with probabilities."""
    rows = []
    for key, count in counter.items():
        outcomes = dict(key)
        rows.append(
            {
                "outcomes": outcomes,
                "count": int(count),
                "prob": float(count / total) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["count"], str(r["outcomes"])))
    return rows


@dataclass
class ProbabilityTableBundle:
    """Cross-referenced profiling artifact consumed by the optimizer."""

    initial: dict[str, Any]
    specialized: dict[str, dict[str, Any]] = field(default_factory=dict)
    wcet_ms: dict[str, float] = field(default_factory=dict)
    runtime_ms: dict[str, float] = field(default_factory=dict)
    threshold_hi: dict[str, float] = field(default_factory=dict)
    kdet: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial": self.initial,
            "specialized": self.specialized,
            "wcet_ms": self.wcet_ms,
            "runtime_ms": self.runtime_ms,
            "threshold_hi": self.threshold_hi,
            "kdet": self.kdet,
            "meta": self.meta,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ProbabilityTableBundle:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            initial=raw["initial"],
            specialized=raw.get("specialized", {}),
            wcet_ms=raw.get("wcet_ms", {}),
            runtime_ms=raw.get("runtime_ms", {}),
            threshold_hi=raw.get("threshold_hi", {}),
            kdet=raw.get("kdet", {}),
            meta=raw.get("meta", {}),
        )


def pr_idk_given_prefix(
    table_rows: list[dict[str, Any]],
    prefix: dict[str, str],
    target_ki: str,
) -> float:
    """Pr(target_ki returns IDK | prefix outcomes fixed)."""
    return cond_pr(table_rows, prefix, target_ki, "IDK")


def cond_pr(
    table_rows: list[dict[str, Any]],
    prefix: dict[str, str],
    target_ki: str,
    outcome: str,
) -> float:
    """
    Paper COND_PR: Pr(Kj returns `outcome` | all Ki in prefix match).

    Denominator: rows where every key in prefix matches.
    Numerator: those rows AND target_ki column equals outcome.
    """
    num = 0.0
    den = 0.0
    for row in table_rows:
        outcomes = row["outcomes"]
        if any(outcomes.get(k) != v for k, v in prefix.items()):
            continue
        if target_ki not in outcomes:
            continue
        p = float(row["prob"])
        den += p
        if outcomes[target_ki] == outcome:
            num += p
    return num / den if den > 0 else 0.0


def prefix_all_idk(classifiers: set[str] | frozenset[str]) -> dict[str, str]:
    """Build prefix dict assuming every Ki in the set returned IDK."""
    return {ki: "IDK" for ki in classifiers}


def cond_pr_specialized(
    table_rows: list[dict[str, Any]],
    prefix: dict[str, str],
    kh: str,
    intermediate_class: str,
    target_ki: str,
    outcome: str,
) -> float:
    """
    COND_PR′ for specialized tables (paper Section IV-B).

    Denominator rows must satisfy:
      - prefix Ki all match (typically IDK)
      - Kh column equals intermediate_class (Kh correctly routed I_l)
    """
    num = 0.0
    den = 0.0
    for row in table_rows:
        outcomes = row["outcomes"]
        if outcomes.get(kh) != intermediate_class:
            continue
        if any(outcomes.get(k) != v for k, v in prefix.items()):
            continue
        if target_ki not in outcomes:
            continue
        p = float(row["prob"])
        den += p
        if outcomes[target_ki] == outcome:
            num += p
    return num / den if den > 0 else 0.0
