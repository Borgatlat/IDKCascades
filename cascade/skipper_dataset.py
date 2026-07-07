"""Build per-position skip training datasets from the empirical outcomes table.

Each "position" in the cascade is a point where a classifier said IDK and
the next classifier in the static order has not yet run. The skip model at
that position decides:
  0 = continue (run the next classifier in the static order)
  1 = skip 1 classifier forward
  2 = skip 2 classifiers forward
  ... etc.
  N = skip to detector (last label, always == number of remaining classifiers)

Ground truth label for sample s at position p:
  Find the first downstream classifier (from the static order starting at p+1)
  whose `accepted[s]` is True in the outcome table. The label is the index of
  that classifier in the remaining chain (0 = next, 1 = one after, etc.). If
  none accept, label = len(remaining chain) = "skip to detector".

Feature construction:
  Initial chain positions (after an initial-chain classifier says IDK):
    [confidence, entropy, margin]  -- 3 features from the IDK classifier's
    softmax output (same as Paper 2).

  Specialized chain positions (after a specialized classifier says IDK):
    [conf_spec, ent_spec, margin_spec,     -- 3 features from the specialized
                                              classifier that said IDK
     conf_id, ent_id, margin_id,           -- 3 features from the identifier
                                              that routed to this group
     group_size]                            -- 1 feature: number of sub-classes
                                              in this group
    = 7 features total.

  The identifier features matter because a barely-confident routing
  (identifier softmax 0.61 at threshold 0.60) means the specialized chain is
  operating on a noisily-assigned sample; skipping further is riskier. A very
  confident routing (0.97) means the specialized chain has clean input.

One dataset file is saved per transition position. At runtime only one skip
model is ever called per transition, so inference overhead is one tiny RF/MLP
call per IDK -- O(0.01-0.1ms), negligible vs. classifier costs of 2-13ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes
from cascade.hierarchy_optimizer import (
    PAPER_DETECTOR_COST_MS,
    Cascade,
    HierarchyOptimizer,
    optimize_empirical_hierarchy,
)

DATASET_DIR = Path("checkpoints/skipper_datasets")

INITIAL_FEATURE_NAMES = ["confidence", "entropy", "margin"]
SPECIALIZED_FEATURE_NAMES = [
    "spec_confidence", "spec_entropy", "spec_margin",
    "id_confidence", "id_entropy", "id_margin",
    "group_size",
]


def _softmax_stats(confidence_col: np.ndarray, outcome_rows: pd.DataFrame) -> np.ndarray:
    """Compute confidence/entropy/margin from the stored per-sample confidence
    value.  We only stored the winning probability (max), so we can compute
    confidence and margin from the softmax output stored in `confidence`.
    For entropy we approximate using the stored confidence as p_max and
    distribute the rest uniformly across (n_classes - 1) bins -- this is an
    approximation, but it has the correct ordering property (higher confidence
    => lower entropy) and avoids re-running the model.

    In practice, Paper 2 found confidence alone was the most predictive single
    feature, and margin/entropy add relatively small increments; this
    approximation is sufficient for a skip decision that already operates on
    noisy signal.

    Parameters
    ----------
    confidence_col : np.ndarray, shape (N,)
        Max softmax probability for each sample.
    outcome_rows : pd.DataFrame
        The rows from the outcomes table for this classifier and these samples.
        Currently unused beyond `confidence_col` but kept for future extension
        (e.g. storing full softmax or top-2 probability for exact margin).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Columns: [confidence, entropy, margin].
    """
    conf = confidence_col
    n_classes = max(2, int(round(1.0 / max(float(1 - conf.mean()), 1e-6))))
    p_rest = np.clip((1.0 - conf) / max(n_classes - 1, 1), 0.0, 1.0)
    eps = 1e-12
    entropy = -(conf * np.log2(conf + eps) + (n_classes - 1) * p_rest * np.log2(p_rest + eps))
    margin = conf - p_rest  # max_prob - second_max_prob (approximated)
    return np.column_stack([conf, entropy, margin])


@dataclass
class PositionDataset:
    """Training data for one skip-decision position."""
    position_id: str       # human-readable key, e.g. "initial_K0" or "spec_K0_suv_K4"
    chain_type: str        # "initial" or "specialized"
    idk_classifier: str    # classifier that said IDK triggering this decision
    remaining: list[str]   # classifiers still available AFTER the IDK one (excl. detector)
    n_choices: int         # len(remaining) + 1, or 2 if is_binary
    is_binary: bool        # True: label space is {0=continue, 1=skip-to-det} regardless
                            # of how many classifiers remain. False: label value IS the
                            # skip distance (0=continue, 1=skip1, ..., len(remaining)=det).
    X: np.ndarray          # shape (N, n_features)
    y: np.ndarray          # shape (N,) integer in [0, n_choices-1]
    feature_names: list[str]
    label_meanings: list[str]  # e.g. ["continue->K3", "skip->K2", "skip->detector"]


def _label_meanings(remaining: list[str], detector_label: str = "detector") -> list[str]:
    if not remaining:
        return [f"skip->{detector_label}"]
    meanings = [f"continue->{remaining[0]}"]
    for i, cid in enumerate(remaining[1:], start=1):
        meanings.append(f"skip{i}->{cid}")
    meanings.append(f"skip->{detector_label}")
    return meanings


def _ground_truth_label(
    sample_ids: np.ndarray,
    remaining: list[str],
    accepted: dict[str, np.ndarray],
    binary: bool = False,
) -> np.ndarray:
    """For each sample, find the index of the first remaining classifier that
    would have accepted it. If none would, label = len(remaining) (= det).

    If binary=True, collapse to {0 = continue (run the immediate next
    classifier), 1 = skip straight to detector}. This intentionally drops
    the ability to skip to an intermediate classifier (e.g. "skip K3, land
    on K2 directly") -- it only decides whether the WHOLE remaining chain is
    worth attempting before falling to the detector.

    This labeling is CORRECTNESS-ONLY: it ignores classifier costs entirely.
    It will say "continue" any time ANY downstream classifier would
    eventually accept, even if reaching it costs more than just paying for
    Kdet directly. See _cost_aware_ground_truth_label for the alternative
    that accounts for this."""
    if not binary:
        labels = np.full(len(sample_ids), len(remaining), dtype=np.int32)
        for skip_distance, cid in enumerate(remaining):
            would_accept = accepted[cid][sample_ids]
            not_yet_labelled = labels == len(remaining)
            labels = np.where(not_yet_labelled & would_accept, skip_distance, labels)
        return labels

    any_accept = np.zeros(len(sample_ids), dtype=bool)
    for cid in remaining:
        any_accept |= accepted[cid][sample_ids]
    return np.where(any_accept, 0, 1).astype(np.int32)


def _cost_aware_ground_truth_label(
    sample_ids: np.ndarray,
    remaining: list[str],
    accepted: dict[str, np.ndarray],
    costs: dict[str, float],
    detector_cost: float,
    binary: bool = True,
) -> np.ndarray:
    """Cost-aware ground truth: for each sample, compute the REALIZED cost of
    two strategies and pick whichever is cheaper for THAT sample:

      (a) "continue": run remaining classifiers in their static order until
          one accepts, paying each one's cost along the way, falling to
          detector_cost if none accept. Realized cost =
              sum(costs[c] for c in remaining[:k+1]) [+ detector_cost if none accept]
          where k is the index of the first accepting classifier (or all of
          `remaining` if none accept).
      (b) "skip to detector now": pay detector_cost only.

    Label 0 = continue is realized-cheaper for this sample, label 1 = skip
    to detector is realized-cheaper. This differs from the plain
    first-acceptor label in exactly the cases that matter: a sample where
    K3 (7.57ms) would eventually accept but only after also paying for some
    other expensive classifier first might still be cheaper to send straight
    to detector if detector_cost is low enough -- the plain label would
    always say "continue" here since SOMETHING downstream accepts,
    regardless of how expensive reaching it is.

    NOTE: this still only supports binary collapse (continue vs. detector).
    A cost-aware MULTICLASS version (skip to a specific intermediate
    classifier) would need to compare every possible landing point's
    realized cost, not just "first acceptor" -- not implemented here since
    your initial-chain positions are already collapsed to binary for the
    correctness-only case too (see build_initial_position_datasets), and
    that's where cost-awareness matters most given K3's high cost.
    """
    if not binary:
        raise NotImplementedError(
            "Cost-aware multiclass labeling is not implemented -- see docstring."
        )

    n = len(sample_ids)
    continue_cost = np.zeros(n, dtype=np.float64)
    resolved = np.zeros(n, dtype=bool)

    for cid in remaining:
        would_accept = accepted[cid][sample_ids]
        c = float(costs.get(cid, 0.0))
        # Every unresolved sample pays this classifier's cost, whether it
        # accepts or not (you only find out by running it).
        continue_cost = np.where(~resolved, continue_cost + c, continue_cost)
        newly_resolved = (~resolved) & would_accept
        resolved = resolved | newly_resolved

    # Samples never accepted by anything in `remaining` also pay detector_cost
    # at the end of the chain.
    continue_cost = np.where(~resolved, continue_cost + detector_cost, continue_cost)

    skip_cost = np.full(n, float(detector_cost), dtype=np.float64)

    # Label 1 (skip to detector) wins on ties to match the optimizer's own
    # tie-breaking (hierarchy_optimizer.py prefers detector when costs are
    # exactly equal -- see `if cost < best_cost` strict-inequality checks).
    return np.where(continue_cost < skip_cost, 0, 1).astype(np.int32)


def build_initial_position_datasets(
    optimizer: HierarchyOptimizer,
    cascade: Cascade,
    payload: dict,
    binary: bool = True,
    cost_aware: bool = False,
) -> list[PositionDataset]:
    """Build one PositionDataset per IDK-able position in the initial chain.

    The initial chain is e.g. [K0, K3, K2, K1, detector]. After K0 IDKs the
    remaining chain is [K3, K2, K1]; after K3 IDKs remaining is [K2, K1]; etc.
    If only the detector is left (after K1 IDKs) there is nothing to skip to,
    so no dataset is produced for that position.

    cost_aware=True uses _cost_aware_ground_truth_label (realized per-sample
    cost comparison between "continue" and "skip to detector") instead of
    the plain first-acceptor label. Forces binary=True since cost-aware
    multiclass labeling isn't implemented. Real per-classifier costs come
    from optimizer.candidates (the same costs the optimizer itself used to
    build this cascade), and detector cost from optimizer.detector_cost
    (which reflects whatever detector_mode the cascade was built with --
    "paper" 10000ms or "trained" ~10.85ms -- so labels stay consistent with
    the cascade structure they're being trained against)."""
    if cost_aware and not binary:
        raise ValueError("cost_aware requires binary=True (see _cost_aware_ground_truth_label)")

    labels_df = payload["labels"]
    outcomes = payload["outcomes"]
    sample_ids = labels_df["sample_id"].to_numpy()

    accepted: dict[str, np.ndarray] = {}
    confidence_map: dict[str, np.ndarray] = {}
    for cid, grp in outcomes.groupby("candidate_id", sort=False):
        ordered = grp.sort_values("sample_id")
        accepted[cid] = ordered["accepted"].to_numpy(dtype=bool)
        confidence_map[cid] = ordered["confidence"].to_numpy(dtype=float)

    classifier_costs = {cid: optimizer._cost(cid) for cid in optimizer.initial_ids}
    detector_cost = optimizer.detector_cost

    initial_without_det = [c for c in cascade.initial if c != cascade.detector]
    datasets = []

    eligible_mask = np.ones(len(sample_ids), dtype=bool)  # samples still in play

    for pos, idk_classifier_id in enumerate(initial_without_det):
        remaining_after = initial_without_det[pos + 1:]  # classifiers AFTER this one

        if not remaining_after:
            # Only the detector follows -- no skip decision (it's forced).
            eligible_mask &= ~accepted[idk_classifier_id]
            continue

        # Samples that reached this position (all prior initial classifiers IDK'd)
        # AND this classifier also IDK'd.
        pos_mask = eligible_mask & ~accepted[idk_classifier_id]
        pos_indices = np.where(pos_mask)[0]

        if len(pos_indices) < 2:
            eligible_mask &= ~accepted[idk_classifier_id]
            continue

        X = _softmax_stats(confidence_map[idk_classifier_id][pos_indices], None)
        if cost_aware:
            y = _cost_aware_ground_truth_label(
                pos_indices, remaining_after, accepted, classifier_costs, detector_cost, binary=True
            )
        else:
            y = _ground_truth_label(pos_indices, remaining_after, accepted, binary=binary)

        datasets.append(PositionDataset(
            position_id=f"initial_{idk_classifier_id}",
            chain_type="initial",
            idk_classifier=idk_classifier_id,
            remaining=remaining_after,
            n_choices=2 if binary else len(remaining_after) + 1,
            is_binary=binary,
            X=X,
            y=y,
            feature_names=INITIAL_FEATURE_NAMES,
            label_meanings=["continue", f"skip->detector"] if binary
                            else _label_meanings(remaining_after),
        ))

        # Advance: only samples that IDK'd this classifier remain eligible for
        # the next position.
        eligible_mask &= ~accepted[idk_classifier_id]

    return datasets


def build_specialized_position_datasets(
    optimizer: HierarchyOptimizer,
    cascade: Cascade,
    payload: dict,
    binary: bool = False,
    cost_aware: bool = False,
) -> list[PositionDataset]:
    """Build one PositionDataset per IDK-able position in each specialized chain.

    For each (router_id, group) pair in cascade.specialized, the specialized
    chain is e.g. [K4, K3, K2, detector] for K0/suv. After K4 IDKs, remaining
    is [K3, K2]. The 7-feature vector includes the specialized classifier's
    softmax stats PLUS the routing identifier's softmax stats and group size.

    cost_aware=True: see build_initial_position_datasets docstring -- same
    realized-cost comparison, forces binary=True. Costs come from
    optimizer._cost(), which looks up ANY candidate (global/specialized/
    identifier) by id, so K3/K2 reused inside specialized chains resolve to
    the same real costs the optimizer itself used.
    """
    if cost_aware and not binary:
        raise ValueError("cost_aware requires binary=True (see _cost_aware_ground_truth_label)")

    labels_df = payload["labels"]
    outcomes = payload["outcomes"]
    sample_ids = labels_df["sample_id"].to_numpy()

    accepted: dict[str, np.ndarray] = {}
    confidence_map: dict[str, np.ndarray] = {}
    prediction_map: dict[str, np.ndarray] = {}
    for cid, grp in outcomes.groupby("candidate_id", sort=False):
        ordered = grp.sort_values("sample_id")
        accepted[cid] = ordered["accepted"].to_numpy(dtype=bool)
        confidence_map[cid] = ordered["confidence"].to_numpy(dtype=float)
        prediction_map[cid] = ordered["prediction"].to_numpy(dtype=int)

    group_sizes: dict[str, int] = {}
    for group in optimizer.groups:
        group_sizes[group] = len([
            c for c in optimizer.specialized_by_group.get(group, ())
        ])

    detector_cost = optimizer.detector_cost

    datasets = []

    for (router_id, group), chain in cascade.specialized.items():
        chain_without_det = [c for c in chain if c != cascade.detector]

        if not chain_without_det:
            continue

        group_idx = optimizer._group_to_intermediate_idx.get(group)
        if group_idx is None:
            continue

        # Samples routed to this group by this router
        router_routed = (
            accepted[router_id]
            & (prediction_map[router_id] == group_idx)
        )
        id_conf = confidence_map[router_id]
        id_stats = _softmax_stats(id_conf, None)  # (N_all, 3), indexed by sample_id

        g_size = float(group_sizes.get(group, 1))
        eligible_mask = router_routed.copy()

        for pos, idk_classifier_id in enumerate(chain_without_det):
            remaining_after = chain_without_det[pos + 1:]

            if not remaining_after:
                eligible_mask &= ~accepted[idk_classifier_id]
                continue

            pos_mask = eligible_mask & ~accepted[idk_classifier_id]
            pos_indices = np.where(pos_mask)[0]

            if len(pos_indices) < 2:
                eligible_mask &= ~accepted[idk_classifier_id]
                continue

            spec_stats = _softmax_stats(confidence_map[idk_classifier_id][pos_indices], None)
            id_stats_here = id_stats[pos_indices]  # router stats for these samples
            group_size_col = np.full((len(pos_indices), 1), g_size)

            X = np.concatenate([spec_stats, id_stats_here, group_size_col], axis=1)
            if cost_aware:
                classifier_costs = {cid: optimizer._cost(cid) for cid in remaining_after}
                y = _cost_aware_ground_truth_label(
                    pos_indices, remaining_after, accepted, classifier_costs, detector_cost, binary=True
                )
            else:
                y = _ground_truth_label(pos_indices, remaining_after, accepted, binary=binary)

            datasets.append(PositionDataset(
                position_id=f"spec_{router_id}_{group}_{idk_classifier_id}",
                chain_type="specialized",
                idk_classifier=idk_classifier_id,
                remaining=remaining_after,
                n_choices=2 if binary else len(remaining_after) + 1,
                is_binary=binary,
                X=X,
                y=y,
                feature_names=SPECIALIZED_FEATURE_NAMES,
                label_meanings=["continue", "skip->detector"] if binary
                                else _label_meanings(remaining_after),
            ))

            eligible_mask &= ~accepted[idk_classifier_id]

    return datasets


def build_all_position_datasets(
    outcomes_path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    save_dir: str | Path = DATASET_DIR,
    binary_initial: bool = True,
    binary_specialized: bool = False,
    cost_aware: bool = False,
) -> dict[str, PositionDataset]:
    """Build and optionally save all skip-decision datasets.

    binary_initial defaults to True: the initial chain (K0/K3/K2/K1) has very
    few skip-1/skip-2 examples relative to continue/skip-to-det (see e.g.
    initial_K0: 70 skip1 + 17 skip2 vs 1652 continue + 711 det), so the
    multiclass model overfits / guesses on those minority classes. Binary
    avoids that failure mode at the cost of not being able to jump straight
    past an intermediate classifier in the initial chain.

    binary_specialized defaults to False: specialized positions had more
    balanced 3-way splits in practice (e.g. spec_K0_coupe_K6: 58/8/34%),
    making the multiclass model more trainable there.

    cost_aware=True: use realized per-sample cost comparison (continue vs.
    skip-to-detector) instead of plain first-acceptor correctness as ground
    truth. Forces binary_initial=True and binary_specialized=True (cost-aware
    multiclass isn't implemented -- see _cost_aware_ground_truth_label).
    Use this when the plain correctness-only labels produce a skip model
    that's net-negative on runtime despite high label accuracy: that means
    the model is "correct" about acceptance but blind to HOW EXPENSIVE
    reaching that acceptance is, which is exactly what cost-aware fixes.

    Returns a dict keyed by position_id.
    """
    if cost_aware:
        binary_initial = True
        binary_specialized = True

    payload = load_empirical_outcomes(outcomes_path)
    optimizer, cascade = optimize_empirical_hierarchy(outcomes_path, detector_mode, detector_cost_ms)

    initial_ds = build_initial_position_datasets(
        optimizer, cascade, payload, binary=binary_initial, cost_aware=cost_aware
    )
    specialized_ds = build_specialized_position_datasets(
        optimizer, cascade, payload, binary=binary_specialized, cost_aware=cost_aware
    )
    all_ds = {ds.position_id: ds for ds in initial_ds + specialized_ds}

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for pos_id, ds in all_ds.items():
        out = save_dir / f"{pos_id}.npz"
        np.savez(
            out,
            X=ds.X,
            y=ds.y,
            feature_names=np.array(ds.feature_names),
            label_meanings=np.array(ds.label_meanings),
            remaining=np.array(ds.remaining),
            n_choices=np.array([ds.n_choices]),
            is_binary=np.array([ds.is_binary]),
            chain_type=np.array([ds.chain_type]),
            idk_classifier=np.array([ds.idk_classifier]),
        )
        print(
            f"  {pos_id}: {len(ds.X)} samples, "
            f"{ds.n_choices} classes, features={ds.X.shape[1]}, "
            f"binary={ds.is_binary}"
        )

    print(f"\nSaved {len(all_ds)} position datasets -> {save_dir}")
    return all_ds


def load_position_dataset(path: str | Path) -> PositionDataset:
    data = np.load(path, allow_pickle=True)
    return PositionDataset(
        position_id=Path(path).stem,
        chain_type=str(data["chain_type"][0]),
        idk_classifier=str(data["idk_classifier"][0]),
        remaining=list(data["remaining"]),
        n_choices=int(data["n_choices"][0]),
        is_binary=bool(data["is_binary"][0]) if "is_binary" in data else (int(data["n_choices"][0]) == 2),
        X=data["X"],
        y=data["y"],
        feature_names=list(data["feature_names"]),
        label_meanings=list(data["label_meanings"]),
    )


def load_all_position_datasets(
    save_dir: str | Path = DATASET_DIR,
) -> dict[str, PositionDataset]:
    save_dir = Path(save_dir)
    datasets = {}
    for path in sorted(save_dir.glob("*.npz")):
        ds = load_position_dataset(path)
        datasets[ds.position_id] = ds
    return datasets


if __name__ == "__main__":
    print("Building skip-decision datasets from empirical outcomes...")
    all_ds = build_all_position_datasets()
    print("\nSummary:")
    for pos_id, ds in all_ds.items():
        counts = np.bincount(ds.y, minlength=ds.n_choices)
        print(f"  {pos_id}")
        for i, (meaning, count) in enumerate(zip(ds.label_meanings, counts)):
            print(f"    label {i} ({meaning}): {count} samples ({100*count/len(ds.y):.1f}%)")
