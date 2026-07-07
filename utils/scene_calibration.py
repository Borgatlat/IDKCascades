"""Scene-shift calibration drift metrics and WIP publication figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from cascade.eval import evaluate_val_set, summarize_traces
from cascade.executor import ExecutorConfig
from cascade.inference import threshold_for_record
from training.trainer import get_ki_labels
from utils.labels import KI_REGISTRY
from utils.scene_inventory import scene_masks

IDK_KI_NAMES = [f"K{i}" for i in range(7)]

# Publication defaults (aligned with plot_confusion_matrices.py).
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
    }
)


def _parse_sensor_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def load_scene_split(
    inventory_path: Path,
    *,
    calibration_sensors: list[str] | None = None,
    deployment_sensors: list[str] | None = None,
) -> dict[str, Any]:
    """Load recommended or manual calibration vs deployment sensor lists."""
    data = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if calibration_sensors is not None and deployment_sensors is not None:
        split = {
            "strategy": "manual",
            "calibration_sensors": sorted(calibration_sensors),
            "deployment_sensors": sorted(deployment_sensors),
        }
    else:
        split = data.get("recommended_split") or data.get("split_analysis", {}).get("recommended_split", {})
    return {
        "scene_proxy": data.get("scene_proxy", "sensor_id"),
        "calibration_sensors": list(split.get("calibration_sensors", [])),
        "deployment_sensors": list(split.get("deployment_sensors", [])),
        "strategy": split.get("strategy", "unknown"),
    }


def scene_indices(metadata: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Row indices where boolean scene mask is True."""
    return np.where(mask)[0]


def subsample_indices(indices: np.ndarray, max_samples: int | None, seed: int = 42) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def compute_reliability_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin confidence scores and compute per-bin accuracy for reliability diagrams."""
    if len(confidences) == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        bins.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "bin_center": float((lo + hi) / 2.0),
                "accuracy": acc,
                "count": count,
            }
        )
    return bins


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over binned confidence vs accuracy."""
    bins = compute_reliability_bins(confidences, correct, n_bins=n_bins)
    if not bins or len(confidences) == 0:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for b in bins:
        weight = b["count"] / total
        ece += weight * abs(b["bin_center"] - b["accuracy"])
    return float(ece)


@torch.inference_mode()
def _forward_batch(
    model: nn.Module,
    mic_batch: torch.Tensor,
    geo_batch: torch.Tensor | None,
    modality: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (max_conf, pred_idx) for a batch."""
    model.eval()
    mic_b = mic_batch.to(device, non_blocking=True)
    if modality == "mic":
        logits = model(mic_b)
    else:
        assert geo_batch is not None
        geo_b = geo_batch.to(device, non_blocking=True)
        logits = model(mic_b, geo_b)
    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)
    return conf, pred_idx


def split_tune_eval_indices(
    indices: np.ndarray,
    *,
    tune_fraction: float = 0.20,
    tune_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into disjoint tune / eval subsets for deployment recalibration."""
    if len(indices) == 0:
        return indices, indices
    n_tune = max(1, int(round(len(indices) * tune_fraction)))
    if tune_samples is not None:
        n_tune = min(n_tune, tune_samples, len(indices) - 1)
    n_tune = min(n_tune, len(indices) - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    tune_idx = np.sort(perm[:n_tune])
    eval_idx = np.sort(perm[n_tune:])
    return tune_idx, eval_idx


def load_baseline_targets(baseline_path: Path) -> dict[str, float]:
    """Per-Ki calibration-scene P(IDK) targets from Sub-step B JSON."""
    data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    targets: dict[str, float] = {}
    for ki_name in IDK_KI_NAMES:
        p_idk = data["ki_drift"][ki_name]["calibration"].get("p_idk")
        if p_idk is not None:
            targets[ki_name] = float(p_idk)
    return targets


def find_threshold_for_target_p_idk(
    confidences: np.ndarray,
    target_p_idk: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 64,
) -> float:
    """
    Binary search H in [0, 1] so mean(conf < H) ~= target_p_idk.

    P(IDK) increases monotonically with H because more samples fall below the bar.
    """
    if len(confidences) == 0:
        return 0.5
    target = float(np.clip(target_p_idk, 0.0, 1.0))
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        achieved = float(np.mean(confidences < mid))
        if abs(achieved - target) <= tol:
            return float(mid)
        if achieved < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


@torch.inference_mode()
def collect_ki_confidences(
    model: nn.Module,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    ki_name: str,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched forward pass; return max(softmax) per valid sample index."""
    spec = KI_REGISTRY[ki_name]
    all_labels = get_ki_labels(metadata, spec)
    valid_mask = all_labels >= 0
    eval_indices = indices[valid_mask[indices]]
    if len(eval_indices) == 0:
        return np.array([], dtype=np.float64)

    conf_list: list[float] = []
    for start in range(0, len(eval_indices), batch_size):
        batch_idx = eval_indices[start : start + batch_size]
        mic_batch = torch.from_numpy(mic[batch_idx][:, None, :, :].copy())
        geo_batch = (
            torch.from_numpy(geo[batch_idx][:, None, :, :].copy())
            if spec.modality != "mic"
            else None
        )
        conf, _ = _forward_batch(model, mic_batch, geo_batch, spec.modality, device)
        conf_list.extend(conf.cpu().numpy().tolist())
    return np.array(conf_list, dtype=np.float64)


@torch.inference_mode()
def tune_thresholds_all_kis(
    models: dict[str, nn.Module | None],
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata: pd.DataFrame,
    tune_indices: np.ndarray,
    target_p_idk: dict[str, float],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Tune H_i on deployment tune set to match calibration-scene P(IDK) targets."""
    overrides: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for ki_name in IDK_KI_NAMES:
        model = models.get(ki_name)
        if model is None:
            raise ValueError(f"Missing model {ki_name}")
        rec = registry.get(ki_name)
        hi_fixed = threshold_for_record(ki_name, rec.threshold_hi if rec else None)
        assert hi_fixed is not None, f"No threshold H_i for {ki_name}"
        target = target_p_idk.get(ki_name, 0.0)
        confidences = collect_ki_confidences(
            model, mic, geo, metadata, tune_indices, ki_name, device,
            batch_size=batch_size,
        )
        hi_tuned = find_threshold_for_target_p_idk(confidences, target)
        achieved = float(np.mean(confidences < hi_tuned)) if len(confidences) else 0.0
        overrides[ki_name] = hi_tuned
        rows.append(
            {
                "ki": ki_name,
                "H_i_fixed": hi_fixed,
                "H_i_tuned": hi_tuned,
                "target_p_idk": target,
                "achieved_p_idk_tune": achieved,
                "n_tune": int(len(confidences)),
            }
        )
    return overrides, rows


def registry_with_thresholds(registry, overrides: dict[str, float]):
    """Return registry copy with deployment-tuned H_i (delegates to ClassifierRegistry helper)."""
    return registry.with_threshold_overrides(overrides)


@torch.inference_mode()
def profile_ki_scene_group(
    model: nn.Module,
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    ki_name: str,
    device: torch.device,
    *,
    batch_size: int = 64,
    threshold_hi_override: float | None = None,
) -> dict[str, Any]:
    """Profile one Ki on a scene-group index subset with fixed or overridden H_i."""
    spec = KI_REGISTRY[ki_name]
    rec = registry.get(ki_name)
    if threshold_hi_override is not None:
        hi = float(threshold_hi_override)
    else:
        hi = threshold_for_record(ki_name, rec.threshold_hi if rec else None)
    assert hi is not None, f"No threshold H_i for {ki_name}"

    all_labels = get_ki_labels(metadata, spec)
    valid_mask = all_labels >= 0
    eval_indices = indices[valid_mask[indices]]
    if len(eval_indices) == 0:
        return {
            "ki": ki_name,
            "n_eval": 0,
            "threshold_hi": hi,
            "registry_p_idk": getattr(rec, "p_idk", None) if rec else None,
        }

    y_true = all_labels[eval_indices]
    conf_list: list[float] = []
    pred_list: list[int] = []
    idk_list: list[bool] = []

    for start in range(0, len(eval_indices), batch_size):
        batch_idx = eval_indices[start : start + batch_size]
        # KiDataset uses (N, 1, H, W); add channel dim for conv encoders.
        mic_batch = torch.from_numpy(mic[batch_idx][:, None, :, :].copy())
        geo_batch = (
            torch.from_numpy(geo[batch_idx][:, None, :, :].copy())
            if spec.modality != "mic"
            else None
        )
        conf, pred_idx = _forward_batch(model, mic_batch, geo_batch, spec.modality, device)
        conf_np = conf.cpu().numpy()
        pred_np = pred_idx.cpu().numpy()
        idk_np = conf_np < hi
        conf_list.extend(conf_np.tolist())
        pred_list.extend(pred_np.tolist())
        idk_list.extend(idk_np.tolist())

    conf_arr = np.array(conf_list, dtype=np.float64)
    pred_arr = np.array(pred_list, dtype=np.int64)
    idk_arr = np.array(idk_list, dtype=bool)
    correct_arr = pred_arr == y_true

    n_eval = len(eval_indices)
    p_idk = float(idk_arr.mean())
    p_correct = float(correct_arr.mean())
    non_idk = ~idk_arr
    non_idk_accuracy = float(correct_arr[non_idk].mean()) if non_idk.any() else None
    mean_confidence = float(conf_arr.mean())
    mean_confidence_non_idk = float(conf_arr[non_idk].mean()) if non_idk.any() else None

    # ECE on non-IDK answers (where calibration claim applies).
    if non_idk.any():
        ece = compute_ece(conf_arr[non_idk], correct_arr[non_idk])
        reliability_bins = compute_reliability_bins(conf_arr[non_idk], correct_arr[non_idk])
    else:
        ece = 0.0
        reliability_bins = []

    return {
        "ki": ki_name,
        "n_eval": n_eval,
        "threshold_hi": hi,
        "registry_p_idk": getattr(rec, "p_idk", None) if rec else None,
        "p_idk": p_idk,
        "p_correct": p_correct,
        "non_idk_accuracy": non_idk_accuracy,
        "mean_confidence": mean_confidence,
        "mean_confidence_non_idk": mean_confidence_non_idk,
        "ece": ece,
        "reliability_bins": reliability_bins,
    }


def profile_all_kis(
    models: dict[str, nn.Module | None],
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata: pd.DataFrame,
    cal_indices: np.ndarray,
    dep_indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Profile K0–K6 on calibration vs deployment scene groups."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for ki_name in IDK_KI_NAMES:
        model = models.get(ki_name)
        if model is None:
            raise ValueError(f"Missing model {ki_name}")
        hi_override = threshold_overrides.get(ki_name) if threshold_overrides else None
        out[ki_name] = {
            "calibration": profile_ki_scene_group(
                model, registry, mic, geo, metadata, cal_indices, ki_name, device,
                batch_size=batch_size,
                threshold_hi_override=hi_override,
            ),
            "deployment": profile_ki_scene_group(
                model, registry, mic, geo, metadata, dep_indices, ki_name, device,
                batch_size=batch_size,
                threshold_hi_override=hi_override,
            ),
        }
    return out


def profile_kis_on_indices(
    models: dict[str, nn.Module | None],
    registry,
    mic: np.ndarray,
    geo: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Profile each Ki on a single index subset (eval holdout)."""
    out: dict[str, dict[str, Any]] = {}
    for ki_name in IDK_KI_NAMES:
        model = models.get(ki_name)
        if model is None:
            raise ValueError(f"Missing model {ki_name}")
        hi_override = threshold_overrides.get(ki_name) if threshold_overrides else None
        out[ki_name] = profile_ki_scene_group(
            model, registry, mic, geo, metadata, indices, ki_name, device,
            batch_size=batch_size,
            threshold_hi_override=hi_override,
        )
    return out


def recalibration_ki_summary(
    fixed_profiles: dict[str, dict[str, Any]],
    tuned_profiles: dict[str, dict[str, Any]],
    tune_rows: list[dict[str, Any]],
    target_p_idk: dict[str, float],
) -> list[dict[str, Any]]:
    """Flat per-Ki before/after rows for deployment eval holdout."""
    tune_by_ki = {r["ki"]: r for r in tune_rows}
    rows: list[dict[str, Any]] = []
    for ki_name in IDK_KI_NAMES:
        fixed = fixed_profiles[ki_name]
        tuned = tuned_profiles[ki_name]
        tr = tune_by_ki.get(ki_name, {})
        rows.append(
            {
                "ki": ki_name,
                "H_i_fixed": tr.get("H_i_fixed", fixed.get("threshold_hi")),
                "H_i_tuned": tr.get("H_i_tuned"),
                "target_p_idk": target_p_idk.get(ki_name),
                "p_idk_fixed_eval": fixed.get("p_idk"),
                "p_idk_tuned_eval": tuned.get("p_idk"),
                "achieved_p_idk_tune": tr.get("achieved_p_idk_tune"),
                "non_idk_acc_fixed": fixed.get("non_idk_accuracy"),
                "non_idk_acc_tuned": tuned.get("non_idk_accuracy"),
                "n_eval": fixed.get("n_eval"),
            }
        )
    return rows


def ki_drift_summary(ki_drift: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flat per-Ki drift table rows."""
    rows: list[dict[str, Any]] = []
    for ki_name in IDK_KI_NAMES:
        cal = ki_drift[ki_name]["calibration"]
        dep = ki_drift[ki_name]["deployment"]
        p_idk_cal = cal.get("p_idk")
        p_idk_dep = dep.get("p_idk")
        delta = None
        if p_idk_cal is not None and p_idk_dep is not None:
            delta = p_idk_dep - p_idk_cal
        rows.append(
            {
                "ki": ki_name,
                "H_i": cal.get("threshold_hi"),
                "registry_p_idk": cal.get("registry_p_idk"),
                "p_idk_calibration": p_idk_cal,
                "p_idk_deployment": p_idk_dep,
                "p_idk_delta": delta,
                "non_idk_acc_calibration": cal.get("non_idk_accuracy"),
                "non_idk_acc_deployment": dep.get("non_idk_accuracy"),
                "ece_calibration": cal.get("ece"),
                "ece_deployment": dep.get("ece"),
                "n_eval_calibration": cal.get("n_eval"),
                "n_eval_deployment": dep.get("n_eval"),
            }
        )
    return rows


def drift_detected(ki_drift: dict[str, dict[str, dict[str, Any]]], *, tol: float = 0.005) -> bool:
    """True if any Ki shows |p_idk_deploy - p_idk_cal| > tol."""
    for ki_name in IDK_KI_NAMES:
        cal = ki_drift[ki_name]["calibration"].get("p_idk")
        dep = ki_drift[ki_name]["deployment"].get("p_idk")
        if cal is not None and dep is not None and abs(dep - cal) > tol:
            return True
    return False


def compare_cascade_scenes(
    bundle,
    cal_indices: np.ndarray,
    dep_indices: np.ndarray,
    config: ExecutorConfig,
    *,
    registry=None,
) -> dict[str, dict[str, Any]]:
    """Run EXPAND cascade on calibration vs deployment scene indices."""
    reg = registry if registry is not None else bundle.registry
    cal_traces = evaluate_val_set(
        bundle.plan,
        bundle.models,
        reg,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        cal_indices,
        bundle.device,
        config,
    )
    dep_traces = evaluate_val_set(
        bundle.plan,
        bundle.models,
        reg,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        dep_indices,
        bundle.device,
        config,
    )
    return {
        "calibration": summarize_traces(cal_traces),
        "deployment": summarize_traces(dep_traces),
    }


def compare_cascade_on_indices(
    bundle,
    indices: np.ndarray,
    config: ExecutorConfig,
    *,
    registry=None,
) -> dict[str, Any]:
    """Run EXPAND cascade on a single index subset (e.g. deployment eval holdout)."""
    reg = registry if registry is not None else bundle.registry
    traces = evaluate_val_set(
        bundle.plan,
        bundle.models,
        reg,
        bundle.mic,
        bundle.geo,
        bundle.metadata,
        indices,
        bundle.device,
        config,
    )
    return summarize_traces(traces)


def plot_ki_drift_table_png(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Publication-style table: per-Ki calibration drift under fixed H_i."""
    df = pd.DataFrame(rows)
    display = df[
        [
            "ki",
            "H_i",
            "registry_p_idk",
            "p_idk_calibration",
            "p_idk_deployment",
            "p_idk_delta",
            "non_idk_acc_calibration",
            "non_idk_acc_deployment",
        ]
    ].copy()

    for col in display.columns[1:]:
        if col in ("ki",):
            continue
        display[col] = display[col].apply(
            lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.4f}"
        )

    col_labels = [
        "Ki",
        r"$H_i$",
        r"$P(\mathrm{IDK})_{\mathrm{reg}}$",
        r"$P(\mathrm{IDK})_{\mathrm{cal}}$",
        r"$P(\mathrm{IDK})_{\mathrm{dep}}$",
        r"$\Delta P(\mathrm{IDK})$",
        r"Acc$_{\mathrm{cal}}^{\neg\mathrm{IDK}}$",
        r"Acc$_{\mathrm{dep}}^{\neg\mathrm{IDK}}$",
    ]
    n_rows = len(display)
    fig_h = max(4.0, 0.55 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="white")
    ax.axis("off")

    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    widths = [0.06, 0.07, 0.11, 0.11, 0.11, 0.10, 0.12, 0.12]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(widths[col])
        if row == 0:
            cell.set_facecolor("#2c5282")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if row % 2 == 0 else "#ffffff")

    fig.suptitle(
        "Per-$K_i$ Calibration Drift Under Scene Shift (Fixed $H_i$)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)


def plot_idk_rate_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Grouped bar chart: P(IDK) calibration vs deployment per Ki."""
    ki_names = [r["ki"] for r in rows]
    cal = [r.get("p_idk_calibration") or 0.0 for r in rows]
    dep = [r.get("p_idk_deployment") or 0.0 for r in rows]
    reg = [r.get("registry_p_idk") or 0.0 for r in rows]

    x = np.arange(len(ki_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(x - width, cal, width, label="Calibration scenes", color="#3182ce")
    ax.bar(x, dep, width, label="Deployment scene", color="#e53e3e")
    ax.bar(x + width, reg, width, label="Registry (val)", color="#718096", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ki_names)
    ax.set_ylabel(r"$P(\mathrm{IDK})$ at fixed $H_i$")
    ax.set_xlabel("Classifier $K_i$")
    ax.set_title("IDK Rate Drift: Calibration vs Deployment Scenes", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, max(cal + dep + reg) * 1.15 + 0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_non_idk_accuracy_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Grouped bar: non-IDK accuracy calibration vs deployment."""
    ki_names = [r["ki"] for r in rows]
    cal = [r.get("non_idk_acc_calibration") or 0.0 for r in rows]
    dep = [r.get("non_idk_acc_deployment") or 0.0 for r in rows]

    x = np.arange(len(ki_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(x - width / 2, cal, width, label="Calibration scenes", color="#3182ce")
    ax.bar(x + width / 2, dep, width, label="Deployment scene", color="#e53e3e")

    ax.set_xticks(x)
    ax.set_xticklabels(ki_names)
    ax.set_ylabel(r"Accuracy when $K_i \neq \mathrm{IDK}$")
    ax.set_xlabel("Classifier $K_i$")
    ax.set_title("Non-IDK Accuracy Under Scene Shift", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_reliability_ax(ax: plt.Axes, bins: list[dict], *, label: str, color: str, linestyle: str) -> None:
    if not bins:
        return
    centers = [b["bin_center"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    ax.plot(centers, accs, "o", label=label, color=color, linestyle=linestyle, linewidth=2, markersize=6)
    for c, a in zip(centers, accs):
        ax.plot([c, c], [c, a], color=color, linestyle=":", alpha=0.4, linewidth=1)


def plot_reliability_panels(
    ki_drift: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    *,
    ki_names: tuple[str, str] = ("K0", "K3"),
) -> None:
    """1x2 reliability diagrams for representative intermediate + global Ki."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    titles = {"K0": "Intermediate ($K_0$)", "K3": "Global ($K_3$)"}

    for ax, ki in zip(axes, ki_names):
        cal_bins = ki_drift[ki]["calibration"].get("reliability_bins", [])
        dep_bins = ki_drift[ki]["deployment"].get("reliability_bins", [])
        _plot_reliability_ax(ax, cal_bins, label="Calibration scenes", color="#3182ce", linestyle="-")
        _plot_reliability_ax(ax, dep_bins, label="Deployment scene", color="#e53e3e", linestyle="--")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence (max softmax)")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(titles.get(ki, ki), fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Reliability Diagrams Under Scene Shift (Fixed $H_i$)", fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cascade_scene_comparison(cascade_summary: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Grouped bars: cascade accuracy, mean latency, Kdet rate."""
    groups = ["calibration", "deployment"]
    labels = ["Calibration\nscenes", "Deployment\nscene"]
    acc = [cascade_summary[g]["accuracy"] * 100 for g in groups]
    lat = [cascade_summary[g]["latency_ms"]["mean"] for g in groups]
    kdet = [cascade_summary[g]["kdet_rate"] * 100 for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor="white")
    colors = ["#3182ce", "#e53e3e"]

    metrics = [
        (acc, "Accuracy (%)", "EXPAND Cascade Accuracy"),
        (lat, r"Mean latency $\bar{C}$ (ms)", "Mean Latency"),
        (kdet, r"$K_{\mathrm{det}}$ hit rate (%)", r"$K_{\mathrm{det}}$ Invocation Rate"),
    ]
    for ax, (values, ylabel, title) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors, edgecolor="#1a202c", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("End-to-End Cascade Impact Under Scene Shift", fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_all_figures(
    ki_drift: dict[str, dict[str, dict[str, Any]]],
    drift_rows: list[dict[str, Any]],
    cascade_summary: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> dict[str, str]:
    """Write all WIP PNG artifacts; return path map."""
    fig_dir = Path(fig_dir)
    paths = {
        "ki_drift_table": fig_dir / "ki_drift_table.png",
        "idk_rate_comparison": fig_dir / "idk_rate_comparison.png",
        "non_idk_accuracy_comparison": fig_dir / "non_idk_accuracy_comparison.png",
        "reliability_k0_k3": fig_dir / "reliability_k0_k3.png",
        "cascade_scene_comparison": fig_dir / "cascade_scene_comparison.png",
    }
    plot_ki_drift_table_png(drift_rows, paths["ki_drift_table"])
    plot_idk_rate_comparison(drift_rows, paths["idk_rate_comparison"])
    plot_non_idk_accuracy_comparison(drift_rows, paths["non_idk_accuracy_comparison"])
    plot_reliability_panels(ki_drift, paths["reliability_k0_k3"])
    plot_cascade_scene_comparison(cascade_summary, paths["cascade_scene_comparison"])
    return {k: str(v) for k, v in paths.items()}


def plot_threshold_shift_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Table: fixed H_i vs tuned H_i per Ki."""
    display = pd.DataFrame(rows)[["ki", "H_i_fixed", "H_i_tuned", "target_p_idk", "achieved_p_idk_tune"]].copy()
    for col in display.columns[1:]:
        display[col] = display[col].apply(lambda v: "—" if v is None else f"{float(v):.4f}")

    col_labels = ["Ki", r"$H_i^{\mathrm{fixed}}$", r"$H_i^{\mathrm{tuned}}$",
                  r"$P(\mathrm{IDK})_{\mathrm{target}}$", r"$P(\mathrm{IDK})_{\mathrm{tune}}$"]
    n_rows = len(display)
    fig_h = max(3.5, 0.55 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c5282")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if row % 2 == 0 else "#ffffff")
    fig.suptitle("Deployment-Scene Threshold Re-Tuning ($H_i$)", fontweight="bold", y=0.98)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)


def plot_idk_before_after(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Grouped bar: eval P(IDK) fixed vs tuned vs calibration target."""
    ki_names = [r["ki"] for r in rows]
    fixed = [r.get("p_idk_fixed_eval") or 0.0 for r in rows]
    tuned = [r.get("p_idk_tuned_eval") or 0.0 for r in rows]
    target = [r.get("target_p_idk") or 0.0 for r in rows]

    x = np.arange(len(ki_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(x - width, fixed, width, label="Fixed $H_i$ (eval)", color="#718096")
    ax.bar(x, tuned, width, label="Retuned $H_i$ (eval)", color="#38a169")
    ax.bar(x + width, target, width, label="Calibration target", color="#3182ce", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ki_names)
    ax.set_ylabel(r"$P(\mathrm{IDK})$ on deployment eval holdout")
    ax.set_xlabel("Classifier $K_i$")
    ax.set_title("IDK Rate: Before vs After Scene Re-Calibration", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, max(fixed + tuned + target) * 1.15 + 0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.close("all")


def plot_cascade_before_after(
    fixed_summary: dict[str, Any],
    tuned_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Grouped bars: cascade accuracy, mean latency, Kdet rate (fixed vs retuned)."""
    labels = ["Fixed $H_i$", "Retuned $H_i$"]
    acc = [fixed_summary["accuracy"] * 100, tuned_summary["accuracy"] * 100]
    lat = [fixed_summary["latency_ms"]["mean"], tuned_summary["latency_ms"]["mean"]]
    kdet = [fixed_summary["kdet_rate"] * 100, tuned_summary["kdet_rate"] * 100]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), facecolor="white")
    colors = ["#718096", "#38a169"]
    metrics = [
        (acc, "Accuracy (%)", "EXPAND Cascade Accuracy"),
        (lat, r"Mean latency $\bar{C}$ (ms)", "Mean Latency"),
        (kdet, r"$K_{\mathrm{det}}$ hit rate (%)", r"$K_{\mathrm{det}}$ Invocation Rate"),
    ]
    for ax, (values, ylabel, title) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors, edgecolor="#1a202c", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Cascade Impact: Deployment Eval Holdout", fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.close("all")


def write_recalibration_figures(
    tune_rows: list[dict[str, Any]],
    ki_rows: list[dict[str, Any]],
    cascade_fixed: dict[str, Any],
    cascade_tuned: dict[str, Any],
    fig_dir: Path,
) -> dict[str, str]:
    """Write Sub-step C PNG artifacts; return path map."""
    fig_dir = Path(fig_dir)
    paths = {
        "threshold_shift_table": fig_dir / "threshold_shift_table.png",
        "idk_before_after": fig_dir / "idk_before_after.png",
        "cascade_before_after": fig_dir / "cascade_before_after.png",
    }
    plot_threshold_shift_table(tune_rows, paths["threshold_shift_table"])
    plot_idk_before_after(ki_rows, paths["idk_before_after"])
    plot_cascade_before_after(cascade_fixed, cascade_tuned, paths["cascade_before_after"])
    return {k: str(v) for k, v in paths.items()}


def plot_scene_calibration_paper_figure(
    baseline_path: Path,
    recalibration_path: Path,
    output_path: Path,
) -> Path:
    """
    Single multi-panel PNG for the WIP paper: scene-shift drift (B) + re-tuning (C).

    Panels:
      (a) Per-Ki P(IDK) on calibration vs deployment scenes (fixed H_i).
      (b) Per-Ki P(IDK) on deployment eval holdout: fixed vs retuned H_i.
      (c) End-to-end cascade impact on eval holdout.
    """
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    recal = json.loads(Path(recalibration_path).read_text(encoding="utf-8"))
    drift_rows = baseline.get("ki_drift_summary") or []
    ki_rows = recal.get("ki_summary") or []
    cascade_fixed = recal["cascade_eval_fixed"]
    cascade_tuned = recal["cascade_eval_tuned"]

    fig = plt.figure(figsize=(12, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.28)

    # --- (a) Phase B: calibration drift ---
    ax_a = fig.add_subplot(gs[0, 0])
    ki_names = [r["ki"] for r in drift_rows]
    x = np.arange(len(ki_names))
    w = 0.35
    p_cal = [r.get("p_idk_calibration") or 0.0 for r in drift_rows]
    p_dep = [r.get("p_idk_deployment") or 0.0 for r in drift_rows]
    ax_a.bar(x - w / 2, p_cal, w, label="Calibration scenes (rs2–rs8)", color="#3182ce")
    ax_a.bar(x + w / 2, p_dep, w, label="Deployment scene (rs1)", color="#e53e3e")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ki_names)
    ax_a.set_ylabel(r"$P(\mathrm{IDK})$ at fixed $H_i$")
    ax_a.set_title("(a) Scene-shift IDK drift (Sub-step B)", fontweight="bold", loc="left")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.set_ylim(0, min(1.0, max(p_cal + p_dep) * 1.12 + 0.04))
    ax_a.grid(axis="y", alpha=0.3)
    # Highlight K1 — largest drift
    k1_i = ki_names.index("K1") if "K1" in ki_names else None
    if k1_i is not None:
        ax_a.annotate(
            r"$\Delta=+0.055$",
            xy=(k1_i + w / 2, p_dep[k1_i]),
            xytext=(k1_i + 0.9, p_dep[k1_i] + 0.06),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.0),
            color="#c53030",
        )

    # --- (b) Phase C: before / after re-tuning on eval holdout ---
    ax_b = fig.add_subplot(gs[0, 1])
    ki_names_c = [r["ki"] for r in ki_rows]
    x_c = np.arange(len(ki_names_c))
    w3 = 0.25
    fixed = [r.get("p_idk_fixed_eval") or 0.0 for r in ki_rows]
    tuned = [r.get("p_idk_tuned_eval") or 0.0 for r in ki_rows]
    target = [r.get("target_p_idk") or 0.0 for r in ki_rows]
    ax_b.bar(x_c - w3, fixed, w3, label="Fixed $H_i$", color="#718096")
    ax_b.bar(x_c, tuned, w3, label="Retuned $H_i$", color="#38a169")
    ax_b.bar(x_c + w3, target, w3, label="Calibration target", color="#3182ce", alpha=0.75)
    ax_b.set_xticks(x_c)
    ax_b.set_xticklabels(ki_names_c)
    ax_b.set_ylabel(r"$P(\mathrm{IDK})$ on eval holdout")
    ax_b.set_title("(b) After deployment-scene re-tuning (Sub-step C)", fontweight="bold", loc="left")
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.set_ylim(0, min(1.0, max(fixed + tuned + target) * 1.12 + 0.04))
    ax_b.grid(axis="y", alpha=0.3)

    # --- (c) Cascade metrics: fixed vs retuned ---
    labels = ["Fixed $H_i$", "Retuned $H_i$"]
    colors = ["#718096", "#38a169"]
    acc = [cascade_fixed["accuracy"] * 100, cascade_tuned["accuracy"] * 100]
    kdet = [cascade_fixed["kdet_rate"] * 100, cascade_tuned["kdet_rate"] * 100]
    lat = [cascade_fixed["latency_ms"]["mean"], cascade_tuned["latency_ms"]["mean"]]

    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    for ax, values, ylabel, title in [
        (ax_c1, acc, "Accuracy (%)", "Accuracy"),
        (ax_c2, kdet, r"$K_{\mathrm{det}}$ rate (%)", r"$K_{\mathrm{det}}$ invocation"),
    ]:
        bars = ax.bar(labels, values, color=colors, edgecolor="#1a202c", linewidth=0.6, width=0.55)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            fmt = f"{val:.2f}" if val < 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt,
                    ha="center", va="bottom", fontsize=9)

    # Latency as inset text block under panel (c) title spanning bottom row
    fig.text(
        0.5, 0.02,
        rf"(c) EXPAND cascade on rs1 eval holdout ($n={cascade_fixed['n']:,}$): "
        rf"$\bar{{C}}$ = {lat[0]:.0f} ms $\rightarrow$ {lat[1]:.0f} ms "
        rf"($\Delta$ = {lat[1] - lat[0]:+.0f} ms)",
        ha="center", fontsize=9, color="#2d3748",
    )
    ax_c1.set_title("(c) Cascade impact", fontweight="bold", loc="left")

    cal_sensors = ", ".join(baseline.get("scene_split", {}).get("calibration_sensors", []))
    dep_sensors = ", ".join(baseline.get("scene_split", {}).get("deployment_sensors", []))
    fig.suptitle(
        "Calibration Maintenance Under M3N-VC Scene Shift",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.94,
        f"Scene proxy: sensor_id  |  Calibration: {cal_sensors}  |  Deployment: {dep_sensors}",
        ha="center", fontsize=9, color="#4a5568",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)
    return output_path


def build_scene_index_sets(
    metadata: pd.DataFrame,
    calibration_sensors: list[str],
    deployment_sensors: list[str],
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cal/dep indices and masks from sensor lists."""
    cal_mask, dep_mask = scene_masks(metadata, calibration_sensors, deployment_sensors)
    cal_idx = subsample_indices(scene_indices(metadata, cal_mask), max_samples, seed=seed)
    dep_idx = subsample_indices(scene_indices(metadata, dep_mask), max_samples, seed=seed + 1)
    return cal_idx, dep_idx, cal_mask, dep_mask
