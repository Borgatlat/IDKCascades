"""Runtime hierarchical IDK cascade with position-specific skip decisions.

At each IDK transition the cascade loads the pre-trained skip model for that
position (loaded once at startup, then cached) and asks: should we continue
to the next classifier, skip N classifiers forward, or jump to the detector?

Skip mode can be "none" (baseline), "rf", or "mlp". The skip overhead per
transition is ~0.05-0.15ms (RF) or ~0.005-0.01ms (MLP), vs. 2-13ms per
classifier -- well under 5% overhead even in the worst case.

Usage
-----
    python -m cascade.runtime_cascade            # baseline, no skipping
    python -m cascade.runtime_cascade --mode rf
    python -m cascade.runtime_cascade --mode mlp
    python -m cascade.runtime_cascade --mode rf --max-samples 500
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cascade.empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes
from cascade.hierarchy_optimizer import (
    PAPER_DETECTOR_COST_MS,
    Cascade,
    HierarchyOptimizer,
    optimize_empirical_hierarchy,
)
from cascade.loader import load_cascade_models
from cascade.skipper_dataset import SPECIALIZED_FEATURE_NAMES, _softmax_stats
from cascade.train_skippers import MLP_SKIPPER_DIR, RF_SKIPPER_DIR
from training.trainer import KiDataset, load_spectrogram_cache
from utils.classifier_registry import ClassifierRegistry
from utils.labels import (
    GLOBAL_CLASS_NAMES,
    INTERMEDIATE_CLASS_NAMES,
    KI_REGISTRY,
    is_deterministic_ki,
    threshold_hi_for_ki,
)

DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REGISTRY_PATH = Path("checkpoints/classifier_registry.json")


# ---------------------------------------------------------------------------
# Skip model loader/caller
# ---------------------------------------------------------------------------

class _SkipModel:
    """Wraps one trained skip model (RF or MLP) for a single position."""

    def __init__(self, path: Path, mode: str):
        self.mode = mode
        self.position_id = path.stem
        if mode == "rf":
            with open(path, "rb") as f:
                self.payload = pickle.load(f)
            self.model = self.payload["model"]
        else:
            # weights_only=False: this checkpoint is our own locally-trained
            # skip model (not a downloaded/untrusted file), and the payload
            # intentionally carries numpy arrays (feature_mean/feature_std)
            # alongside the state_dict, which PyTorch's default weights_only=True
            # (as of torch 2.6+) rejects.
            self.payload = torch.load(path, map_location="cpu", weights_only=False)
            p = self.payload
            self.model = nn.Sequential(
                nn.Linear(p["input_size"], p["hidden_size"]),
                nn.ReLU(),
                nn.Dropout(0.0),  # no dropout at inference
                nn.Linear(p["hidden_size"], p["output_size"]),
            )
            self.model.load_state_dict(p["model_state_dict"])
            self.model.eval()

        self.remaining = list(self.payload["remaining"])
        self.n_choices = int(self.payload["n_choices"])
        self.label_meanings = list(self.payload["label_meanings"])
        # Read explicitly from the saved payload (set at dataset-build time)
        # rather than inferring from n_choices/len(remaining) shape, which is
        # ambiguous: a multiclass position with exactly 1 remaining classifier
        # also has n_choices==2, indistinguishable from a true binary model
        # by shape alone.
        self.is_binary = bool(self.payload.get("is_binary", self.n_choices == 2))

        # MLP was trained on standardized features (see train_skippers.py) --
        # without applying the SAME mean/std at inference, raw features (e.g.
        # group_size as a raw int next to confidence in [0,1]) would be wildly
        # out of the distribution the model was trained on. RF needs no such
        # transform since tree splits are scale-invariant.
        if mode == "mlp":
            self.feature_mean = self.payload.get("feature_mean")
            self.feature_std = self.payload.get("feature_std")
        else:
            self.feature_mean = None
            self.feature_std = None

    def decide(self, features: np.ndarray) -> int:
        """Return skip label: 0=continue, 1=skip1, ..., n_choices-1=skip-to-det."""
        x = features.reshape(1, -1)
        if self.mode == "rf":
            return int(self.model.predict(x)[0])
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x.astype(np.float32)))
            return int(logits.argmax(dim=1).item())


def _load_skip_models(mode: str) -> dict[str, _SkipModel]:
    if mode == "none":
        return {}
    skip_dir = RF_SKIPPER_DIR if mode == "rf" else MLP_SKIPPER_DIR
    ext = "*.pkl" if mode == "rf" else "*.pt"
    models = {}
    for path in sorted(Path(skip_dir).glob(ext)):
        m = _SkipModel(path, mode)
        models[m.position_id] = m
    return models


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    total: int = 0
    correct: int = 0
    total_runtime_s: float = 0.0
    classifier_runs: dict = field(default_factory=dict)
    skip_decisions: dict = field(default_factory=dict)  # position_id -> {continue, skip_N, skip_det}
    skips_by_distance: list = field(default_factory=lambda: [0, 0, 0, 0, 0])  # index = skip distance

    def record_skip(self, position_id: str, label: int, n_choices: int, label_meanings: list):
        if position_id not in self.skip_decisions:
            self.skip_decisions[position_id] = {}
        meaning = label_meanings[label] if label < len(label_meanings) else f"label_{label}"
        self.skip_decisions[position_id][meaning] = \
            self.skip_decisions[position_id].get(meaning, 0) + 1
        dist = min(label, len(self.skips_by_distance) - 1)
        self.skips_by_distance[dist] += 1

    def record_classifier(self, name: str, elapsed: float, correct: bool | None = None):
        if name not in self.classifier_runs:
            self.classifier_runs[name] = {"runs": 0, "runtime_s": 0.0, "ends": 0}
        self.classifier_runs[name]["runs"] += 1
        self.classifier_runs[name]["runtime_s"] += elapsed
        self.total_runtime_s += elapsed

    def summary(self) -> dict:
        return {
            "accuracy": self.correct / self.total if self.total else 0.0,
            "average_runtime_ms": 1000 * self.total_runtime_s / self.total if self.total else 0.0,
            "total": self.total,
            "correct": self.correct,
            "classifier_runs": self.classifier_runs,
            "skip_decisions": self.skip_decisions,
            "skips_by_distance": self.skips_by_distance,
        }


# ---------------------------------------------------------------------------
# Runtime cascade
# ---------------------------------------------------------------------------

class RuntimeCascade:
    """Runs the synthesized cascade over one sample at a time, with optional
    per-position skip decisions."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        registry: ClassifierRegistry,
        optimizer: HierarchyOptimizer,
        cascade: Cascade,
        labels_df,
        skip_mode: str = "none",
        detector_override_ms: float | None = None,
    ):
        """
        detector_override_ms: if set, _run_detector does NOT call the real
        Kdet model. Instead it sleeps for detector_override_ms milliseconds
        and returns the sample's true label (i.e. "always correct", matching
        the paper's footnote-1 assumption that Kdet never errs). This lets
        you measure what the cascade's timing would look like if Kdet really
        did cost what the optimizer assumed when building the cascade (e.g.
        the paper's 10000ms constant, or a more realistic 100ms test value),
        rather than the real trained Kdet's actual ~10.85ms.

        If None (default), the real Kdet checkpoint is run and its own
        prediction is used (which may be wrong, since the trained Kdet is
        only ~94% accurate per the registry -- not the paper's "always
        correct" assumption).
        """
        self.models = models
        self.registry = registry
        self.optimizer = optimizer
        self.cascade = cascade
        self.skip_mode = skip_mode
        self.skip_models = _load_skip_models(skip_mode)
        self.stats = RunStats()
        self.detector_override_ms = detector_override_ms

        # Build label lookup: ki_name -> threshold
        self.thresholds = {
            ki: threshold_hi_for_ki(ki) for ki in KI_REGISTRY
        }

        # Build class name lists per classifier for shared-label remapping
        self._class_names = {
            ki: registry.get(ki).class_names for ki in KI_REGISTRY if registry.get(ki)
        }
        self._is_intermediate = {
            ki: KI_REGISTRY[ki].level == "intermediate" for ki in KI_REGISTRY
        }

    @torch.inference_mode()
    def _forward(self, ki_name: str, mic: torch.Tensor, geo: torch.Tensor):
        """Run one classifier, return (accepted, shared_prediction, confidence)."""
        spec = KI_REGISTRY[ki_name]
        model = self.models[ki_name]

        if spec.modality == "mic":
            logits = model(mic)
        else:
            logits = model(mic, geo)

        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        confidence = float(conf.item())
        class_idx = int(idx.item())
        class_names = self._class_names.get(ki_name, [])

        if is_deterministic_ki(ki_name):
            accepted = True
        else:
            threshold = self.thresholds.get(ki_name)
            accepted = threshold is not None and confidence >= threshold

        if self._is_intermediate.get(ki_name, False):
            pred = INTERMEDIATE_CLASS_NAMES.index(class_names[class_idx]) \
                if class_names and class_idx < len(class_names) else class_idx
        else:
            pred = GLOBAL_CLASS_NAMES.index(class_names[class_idx]) \
                if class_names and class_idx < len(class_names) else class_idx

        return accepted, pred, confidence

    def _run_with_timing(self, ki_name: str, mic, geo) -> tuple:
        t0 = perf_counter()
        accepted, pred, conf = self._forward(ki_name, mic, geo)
        elapsed = perf_counter() - t0
        self.stats.record_classifier(ki_name, elapsed)
        return accepted, pred, conf

    def _maybe_skip(
        self,
        position_id: str,
        idk_conf: float,
        remaining: list[str],
        id_conf: float | None,
        group_size: int | None,
    ) -> tuple[bool, int]:
        """Call the skip model for this position.

        Returns (go_to_detector, skip_distance):
          - go_to_detector=True: jump straight to the detector, skip_distance
            is meaningless.
          - go_to_detector=False: advance `skip_distance` classifiers forward
            from the immediate next one (0 = run the very next classifier
            normally, i.e. "continue").
        """
        if self.skip_mode == "none" or position_id not in self.skip_models:
            return False, 0  # always continue

        sm = self.skip_models[position_id]
        is_specialized = position_id.startswith("spec_")

        if is_specialized and id_conf is not None:
            id_stats = _softmax_stats(np.array([id_conf]), None)[0]
            spec_stats = _softmax_stats(np.array([idk_conf]), None)[0]
            g_size = float(group_size or 1)
            features = np.concatenate([spec_stats, id_stats, [g_size]])
        else:
            features = _softmax_stats(np.array([idk_conf]), None)[0]

        label = sm.decide(features)
        self.stats.record_skip(position_id, label, sm.n_choices, sm.label_meanings)

        if sm.is_binary:
            # label 0 = continue (run next classifier normally), label 1 =
            # skip straight to detector. This is independent of how many
            # classifiers remain -- it never skips to an intermediate one.
            return (label == 1), 0

        # Multiclass: label value IS the skip distance. label == len(remaining)
        # means "skip to detector" (the last class in the multiclass scheme).
        if label >= len(remaining):
            return True, 0
        return False, label

    def predict(self, mic: torch.Tensor, geo: torch.Tensor, true_label: int) -> int:
        """Run one sample through the cascade. Returns predicted global label index."""
        initial_without_det = [c for c in self.cascade.initial if c != self.cascade.detector]
        i = 0
        id_conf: float | None = None  # confidence of the identifier that routed to a group

        while i < len(initial_without_det):
            ki_name = initial_without_det[i]
            accepted, pred, conf = self._run_with_timing(ki_name, mic, geo)

            if accepted:
                if self._is_intermediate.get(ki_name, False):
                    # Identifier: route to specialized chain
                    group_name = INTERMEDIATE_CLASS_NAMES[pred] \
                        if pred < len(INTERMEDIATE_CLASS_NAMES) else None
                    id_conf = conf
                    chain_key = (ki_name, group_name)
                    chain = self.cascade.specialized.get(chain_key, [self.cascade.detector])
                    return self._run_specialized(chain, mic, geo, id_conf,
                                                  group_name, true_label)
                else:
                    return pred
            else:
                # IDK: consult skip model
                remaining_after = initial_without_det[i + 1:]
                position_id = f"initial_{ki_name}"
                go_to_detector, skip_distance = self._maybe_skip(
                    position_id, conf, remaining_after, None, None
                )
                if go_to_detector:
                    return self._run_detector(mic, geo, true_label)
                i = i + 1 + skip_distance  # advance past this one + skip distance

        return self._run_detector(mic, geo, true_label)

    def _run_specialized(
        self,
        chain: list[str],
        mic, geo,
        id_conf: float,
        group_name: str | None,
        true_label: int,
    ) -> int:
        chain_without_det = [c for c in chain if c != self.cascade.detector]
        group_size = len(self.optimizer.specialized_by_group.get(group_name, ())) \
            if group_name else 1

        # Find the router_id (the identifier that triggered this specialized chain)
        router_id = None
        for (rid, grp), ch in self.cascade.specialized.items():
            if grp == group_name and ch == chain:
                router_id = rid
                break

        i = 0
        while i < len(chain_without_det):
            ki_name = chain_without_det[i]
            accepted, pred, conf = self._run_with_timing(ki_name, mic, geo)
            if accepted:
                return pred
            else:
                remaining_after = chain_without_det[i + 1:]
                position_id = f"spec_{router_id}_{group_name}_{ki_name}"
                go_to_detector, skip_distance = self._maybe_skip(
                    position_id, conf, remaining_after, id_conf, group_size
                )
                if go_to_detector:
                    return self._run_detector(mic, geo, true_label)
                i = i + 1 + skip_distance

        return self._run_detector(mic, geo, true_label)

    def _run_detector(self, mic: torch.Tensor, geo: torch.Tensor, true_label: int = -1) -> int:
        if self.detector_override_ms is not None:
            # Synthetic Kdet: don't run the real model, and don't actually
            # block for detector_override_ms either -- there's no reason to
            # burn real wall-clock time sleeping when only the RECORDED
            # elapsed time matters for the benchmark. We just account for
            # the assumed cost directly. Returns the ground-truth label, per
            # the paper's footnote-1 assumption that Kdet never errs.
            self.stats.record_classifier("Kdet", self.detector_override_ms / 1000.0)
            return true_label

        t0 = perf_counter()
        accepted, pred, conf = self._forward("Kdet", mic, geo)
        elapsed = perf_counter() - t0
        self.stats.record_classifier("Kdet", elapsed)
        return pred


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    skip_mode: str = "none",
    outcomes_path: str | Path = DEFAULT_OUTPUT_PATH,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    detector_runtime_override_ms: float | None = None,
    max_samples: int | None = None,
    batch_size: int = 1,
) -> dict:
    """
    detector_cost_ms: the Kdet cost the OPTIMIZER assumes when deciding which
        cascade to build (see hierarchy_optimizer.py). Only used if
        detector_mode="paper".
    detector_runtime_override_ms: if set, the RUNTIME cascade sleeps this
        many ms instead of calling the real Kdet model, and returns the true
        label (always correct). Keep this consistent with detector_cost_ms
        if you want the optimizer's assumption and the runtime's actual
        behavior to match (e.g. both 10000, or both 100). Leave as None to
        use the real trained Kdet checkpoint at runtime regardless of what
        the optimizer assumed when building the cascade.
    """
    processed_dir = Path(processed_dir)
    checkpoint_dir = Path(checkpoint_dir)

    mic_arr, geo_arr, metadata = load_spectrogram_cache(processed_dir)

    from utils.splits import COUPE_VAL_RUNS, DEFAULT_VAL_RUNS, SUV_VAL_RUNS
    eval_runs = DEFAULT_VAL_RUNS | SUV_VAL_RUNS | COUPE_VAL_RUNS
    mask = metadata["run_id"].astype(str).isin(eval_runs).to_numpy()
    eval_meta = metadata.loc[mask].reset_index(drop=True)

    from utils.labels import GLOBAL_CLASS_NAMES
    true_labels = eval_meta["global_label"].map(
        {name: i for i, name in enumerate(GLOBAL_CLASS_NAMES)}
    ).fillna(-1).to_numpy(dtype=int)

    dummy_labels = np.zeros(int(mask.sum()), dtype=np.int64)
    from training.trainer import KiDataset
    dataset = KiDataset(mic_arr[mask], geo_arr[mask], dummy_labels,
                        modality="both", augment=False)

    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)
    optimizer, cascade = optimize_empirical_hierarchy(outcomes_path, detector_mode, detector_cost_ms)

    runtime_cascade = RuntimeCascade(models, registry, optimizer, cascade,
                                      eval_meta, skip_mode=skip_mode,
                                      detector_override_ms=detector_runtime_override_ms)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    total = correct = 0
    t_start = perf_counter()

    print(f"Running cascade (skip_mode={skip_mode}) over {len(dataset)} samples...")
    for idx, (mic_b, geo_b, _) in enumerate(loader):
        if max_samples is not None and total >= max_samples:
            break

        mic_b = mic_b.to(device, non_blocking=True)
        geo_b = geo_b.to(device, non_blocking=True)
        true_label = int(true_labels[idx]) if idx < len(true_labels) else -1

        # NOTE: total_runtime_s is now accumulated inside RunStats.record_classifier
        # (sum of each classifier's recorded cost for this sample), not wall-clock
        # around predict(). This is necessary because the synthetic Kdet override
        # (detector_runtime_override_ms) doesn't actually block for that duration
        # -- it just records the assumed cost -- so wall-clock around predict()
        # would silently miss it entirely.
        pred = runtime_cascade.predict(mic_b, geo_b, true_label)

        runtime_cascade.stats.total += 1
        if pred == true_label:
            runtime_cascade.stats.correct += 1
        total += 1
        correct += int(pred == true_label)

    summary = runtime_cascade.stats.summary()
    print(f"\nskip_mode={skip_mode}")
    print(f"accuracy:          {summary['accuracy']:.4f}")
    print(f"avg runtime (ms):  {summary['average_runtime_ms']:.3f}")
    print(f"total samples:     {summary['total']}")
    print(f"\nclassifier runs:")
    for name, stats in sorted(summary["classifier_runs"].items()):
        avg_ms = 1000 * stats["runtime_s"] / stats["runs"] if stats["runs"] else 0
        print(f"  {name}: {stats['runs']} runs, avg {avg_ms:.2f}ms")
    if summary["skip_decisions"]:
        print(f"\nskip decisions by position:")
        for pos_id, decisions in sorted(summary["skip_decisions"].items()):
            print(f"  {pos_id}: {decisions}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "rf", "mlp"], default="none")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--detector-mode", choices=["paper", "trained"], default="paper")
    parser.add_argument("--detector-cost-ms", type=float, default=PAPER_DETECTOR_COST_MS,
                        help="Kdet cost the OPTIMIZER assumes when building the cascade "
                             "(only used if --detector-mode=paper)")
    parser.add_argument("--detector-runtime-ms", type=float, default=None,
                        help="If set, skip the real Kdet model at runtime: sleep this many "
                             "ms and return the true label instead. Leave unset to use the "
                             "real trained Kdet checkpoint (~94%% accurate, ~10.85ms).")
    args = parser.parse_args()

    if args.detector_mode == "paper" and args.detector_runtime_ms is not None \
            and args.detector_runtime_ms != args.detector_cost_ms:
        print(f"NOTE: optimizer assumes Kdet costs {args.detector_cost_ms}ms but runtime "
              f"will actually take {args.detector_runtime_ms}ms -- these are intentionally "
              f"different if that's what you're testing, but double check this is on purpose.")

    benchmark(
        skip_mode=args.mode,
        max_samples=args.max_samples,
        detector_mode=args.detector_mode,
        detector_cost_ms=args.detector_cost_ms,
        detector_runtime_override_ms=args.detector_runtime_ms,
    )
