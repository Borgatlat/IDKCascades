"""h24-only inventory: sensor coverage within one M3N-VC scene (h24).

NOTE: For the scene *detector*, a scene is an M3N-VC environment (h24, h08,
s31, a06, i29, i22) — terrain/weather domain — NOT a sensor node (rs1, rs2).
Sensor differences within a scene are negligible for detection. See
docs/scene_detector_plan.md and checkpoints/scene_catalog.json.

This module's sensor_id split was a WIP proxy for h24 drift prototyping only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_data import DEFAULT_H24_DIR, DEFAULT_OUTPUT_DIR, _file_metadata, load_h24_two_second_segments
from utils.labels import metadata_row_labels

SCENE_PROXY = "sensor_id"
SEGMENT_DURATION_S = 2.0
REQUIRED_METADATA_COLS = frozenset({"sensor_id", "run_id", "segment_key"})


def _parse_sensor_list(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def scan_raw_parquet_files(raw_dir: Path) -> pd.DataFrame:
    """Lightweight inventory from raw *_mic.parquet filenames (no waveform read)."""
    raw_dir = Path(raw_dir)
    rows: list[dict[str, str]] = []
    for mic_path in sorted(raw_dir.glob("*_mic.parquet")):
        meta = _file_metadata(mic_path, "_mic")
        rows.append(meta)
    if not rows:
        raise FileNotFoundError(f"No *_mic.parquet files found in {raw_dir}")
    return pd.DataFrame(rows)


def _ensure_label_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add global/intermediate labels from run_id when missing."""
    df = metadata.copy()
    if "global_label" not in df.columns or "intermediate_label" not in df.columns:
        label_rows = [metadata_row_labels(str(rid)) for rid in df["run_id"].astype(str)]
        labels_df = pd.DataFrame(label_rows)
        for col in labels_df.columns:
            if col not in df.columns:
                df[col] = labels_df[col].values
    return df


def _segment_table_from_raw(raw_dir: Path, segment_seconds: float) -> pd.DataFrame:
    """Rebuild segment-level table from mic parquet (no STFT)."""
    mic_segments, _ = load_h24_two_second_segments(
        data_dir=raw_dir,
        segment_seconds=segment_seconds,
    )
    grouped = mic_segments.groupby("segment_id", sort=True).first().reset_index()
    rows: list[dict] = []
    for _, row in grouped.iterrows():
        labels = metadata_row_labels(str(row["run_id"]))
        rows.append(
            {
                "segment_key": f"{row['run_id']}_{row['sensor_id']}_seg{int(row['segment_number']):05d}",
                "run_id": str(row["run_id"]),
                "sensor_id": str(row["sensor_id"]),
                "segment_number": int(row["segment_number"]),
                "source_file": str(row.get("source_file", "")),
                **labels,
            }
        )
    return pd.DataFrame(rows)


def load_segment_table(
    processed_dir: Path | str = DEFAULT_OUTPUT_DIR,
    raw_dir: Path | str = DEFAULT_H24_DIR,
    segment_seconds: float = SEGMENT_DURATION_S,
    *,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load segment metadata; prefer processed parquet, verify against raw file list.

    Returns:
        (metadata DataFrame, list of warning strings)
    """
    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    warnings: list[str] = []
    meta_path = processed_dir / "h24_metadata.parquet"

    raw_files = scan_raw_parquet_files(raw_dir)
    raw_sensor_ids = set(raw_files["sensor_id"].astype(str).unique())
    raw_file_count = len(raw_files)

    metadata: pd.DataFrame | None = None
    source = "unknown"

    if not force_rebuild and meta_path.exists():
        candidate = pd.read_parquet(meta_path)
        missing = REQUIRED_METADATA_COLS - set(candidate.columns)
        if missing:
            warnings.append(
                f"h24_metadata.parquet missing columns {sorted(missing)}; rebuilding from raw."
            )
        else:
            metadata = _ensure_label_columns(candidate)
            source = "processed"

    if metadata is None:
        warnings.append("Building segment table from raw mic parquet (no STFT).")
        metadata = _segment_table_from_raw(raw_dir, segment_seconds)
        source = "raw_rebuild"

    proc_sensors = set(metadata["sensor_id"].astype(str).unique())
    if raw_sensor_ids != proc_sensors:
        only_raw = sorted(raw_sensor_ids - proc_sensors)
        only_proc = sorted(proc_sensors - raw_sensor_ids)
        if only_raw:
            warnings.append(f"Sensors in raw but not metadata: {only_raw}")
        if only_proc:
            warnings.append(f"Sensors in metadata but not raw filenames: {only_proc}")

    if source == "processed" and len(metadata) == 0:
        warnings.append("Processed metadata empty; consider --force-rebuild-table.")

    metadata.attrs["load_source"] = source
    metadata.attrs["raw_file_count"] = raw_file_count
    return metadata, warnings


def _crosstab_to_dict(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Convert crosstab to nested JSON-serializable dict."""
    out: dict[str, dict[str, int]] = {}
    for idx in table.index.astype(str):
        out[idx] = {str(col): int(table.loc[idx, col]) for col in table.columns}
    return out


def _sensor_summary(metadata: pd.DataFrame, sensor_id: str) -> dict[str, Any]:
    sub = metadata[metadata["sensor_id"].astype(str) == sensor_id]
    n = len(sub)
    global_labels = sorted(sub["global_label"].astype(str).unique().tolist())
    intermediate_labels = sorted(sub["intermediate_label"].astype(str).unique().tolist())
    run_ids = sorted(sub["run_id"].astype(str).unique().tolist())

    per_global: dict[str, int] = {}
    if "global_label" in sub.columns:
        per_global = sub["global_label"].astype(str).value_counts().to_dict()

    per_run: dict[str, int] = sub["run_id"].astype(str).value_counts().to_dict()

    min_per_class = min(per_global.values()) if per_global else 0

    return {
        "n_segments": n,
        "duration_hours": round(n * SEGMENT_DURATION_S / 3600.0, 4),
        "run_ids": run_ids,
        "global_labels": global_labels,
        "intermediate_labels": intermediate_labels,
        "segments_per_global_label": {str(k): int(v) for k, v in per_global.items()},
        "segments_per_run_id": {str(k): int(v) for k, v in per_run.items()},
        "min_segments_per_global_class": int(min_per_class),
    }


def build_scene_inventory(metadata: pd.DataFrame) -> dict[str, Any]:
    """Aggregate RTS-relevant scene statistics from segment metadata."""
    df = _ensure_label_columns(metadata)
    sensor_ids = sorted(df["sensor_id"].astype(str).unique().tolist())

    crosstab_global = pd.crosstab(
        df["sensor_id"].astype(str),
        df["global_label"].astype(str),
        dropna=False,
    )
    crosstab_run = pd.crosstab(
        df["sensor_id"].astype(str),
        df["run_id"].astype(str),
        dropna=False,
    )

    sensors = {sid: _sensor_summary(df, sid) for sid in sensor_ids}

    min_per_class_per_sensor = {
        sid: sensors[sid]["min_segments_per_global_class"] for sid in sensor_ids
    }

    return {
        "scene_proxy": SCENE_PROXY,
        "n_segments": int(len(df)),
        "n_sensors": len(sensor_ids),
        "sensor_ids": sensor_ids,
        "duration_hours_total": round(len(df) * SEGMENT_DURATION_S / 3600.0, 4),
        "load_source": metadata.attrs.get("load_source", "unknown"),
        "raw_mic_file_count": metadata.attrs.get("raw_file_count"),
        "sensors": sensors,
        "crosstab_global_label": _crosstab_to_dict(crosstab_global),
        "crosstab_run_id": _crosstab_to_dict(crosstab_run),
        "min_segments_per_class_per_sensor": min_per_class_per_sensor,
    }


def _sensor_eligible(
    sensor_summary: dict[str, Any],
    min_segments: int,
) -> tuple[bool, list[str]]:
    """Check if a sensor has enough data and SUV+COUPE coverage."""
    blockers: list[str] = []
    if sensor_summary["n_segments"] < min_segments:
        blockers.append(
            f"insufficient_segments ({sensor_summary['n_segments']} < {min_segments})"
        )
    inter = set(sensor_summary["intermediate_labels"])
    if "suv" not in inter:
        blockers.append("missing_suv")
    if "coupe" not in inter:
        blockers.append("missing_coupe")
    return len(blockers) == 0, blockers


def _split_feasibility(
    inventory: dict[str, Any],
    calibration_sensors: list[str],
    deployment_sensors: list[str],
) -> tuple[bool, list[str]]:
    """Validate a proposed calibration vs deployment partition."""
    blockers: list[str] = []
    all_sensors = set(inventory["sensor_ids"])

    if len(all_sensors) < 2:
        blockers.append("only_one_sensor")

    cal = set(calibration_sensors)
    dep = set(deployment_sensors)

    if not cal:
        blockers.append("empty_calibration_sensors")
    if not dep:
        blockers.append("empty_deployment_sensors")
    if cal & dep:
        blockers.append("overlap_calibration_deployment")

    unknown = (cal | dep) - all_sensors
    if unknown:
        blockers.append(f"unknown_sensors: {sorted(unknown)}")

    for sid in deployment_sensors:
        if sid not in inventory["sensors"]:
            continue
        s = inventory["sensors"][sid]
        if "background" not in s["global_labels"]:
            blockers.append(f"deploy_sensor_{sid}_missing_background")
        if "suv" not in s["intermediate_labels"]:
            blockers.append(f"deploy_sensor_{sid}_missing_suv")
        if "coupe" not in s["intermediate_labels"]:
            blockers.append(f"deploy_sensor_{sid}_missing_coupe")

    return len(blockers) == 0, blockers


def _score_deploy_candidate(sensor_summary: dict[str, Any]) -> float:
    """Higher = better deployment holdout (class diversity + segment count)."""
    n_global = len(sensor_summary["global_labels"])
    n_inter = len(sensor_summary["intermediate_labels"])
    bg = sensor_summary["segments_per_global_label"].get("background", 0)
    return (
        n_global * 1000
        + n_inter * 100
        + min(sensor_summary["n_segments"], 5000)
        + min(bg, 500)
    )


def recommend_scene_split(
    inventory: dict[str, Any],
    *,
    strategy: str = "leave_one_sensor_out",
    min_segments: int = 100,
    calibration_sensors: list[str] | None = None,
    deployment_sensors: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend calibration vs deployment scene partition."""
    all_sensors = inventory["sensor_ids"]
    sensors = inventory["sensors"]

    eligible: list[str] = []
    ineligible: dict[str, list[str]] = {}
    for sid in all_sensors:
        ok, blockers = _sensor_eligible(sensors[sid], min_segments)
        if ok:
            eligible.append(sid)
        else:
            ineligible[sid] = blockers

    candidates: list[dict[str, Any]] = []

    if calibration_sensors is not None and deployment_sensors is not None:
        feasible, blockers = _split_feasibility(inventory, calibration_sensors, deployment_sensors)
        primary = {
            "strategy": "manual",
            "calibration_sensors": sorted(calibration_sensors),
            "deployment_sensors": sorted(deployment_sensors),
            "feasible_for_shift_experiment": feasible,
            "blockers": blockers,
        }
    elif strategy == "leave_one_sensor_out":
        if len(eligible) < 2:
            primary = {
                "strategy": strategy,
                "calibration_sensors": [],
                "deployment_sensors": [],
                "feasible_for_shift_experiment": False,
                "blockers": ["fewer_than_two_eligible_sensors"],
            }
        else:
            ranked = sorted(eligible, key=lambda s: _score_deploy_candidate(sensors[s]), reverse=True)
            for deploy_sid in ranked:
                cal = sorted(s for s in all_sensors if s != deploy_sid)
                dep = [deploy_sid]
                feasible, blockers = _split_feasibility(inventory, cal, dep)
                entry = {
                    "strategy": strategy,
                    "calibration_sensors": cal,
                    "deployment_sensors": dep,
                    "feasible_for_shift_experiment": feasible,
                    "blockers": blockers,
                    "deploy_score": _score_deploy_candidate(sensors[deploy_sid]),
                }
                candidates.append(entry)

            feasible_candidates = [c for c in candidates if c["feasible_for_shift_experiment"]]
            primary = feasible_candidates[0] if feasible_candidates else (candidates[0] if candidates else {})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not candidates and primary:
        candidates = [primary]

    return {
        "strategy": primary.get("strategy", strategy),
        "min_segments": min_segments,
        "eligible_sensors": eligible,
        "ineligible_sensors": ineligible,
        "recommended_split": primary,
        "candidate_splits": candidates[:10],
    }


def scene_masks(
    metadata: pd.DataFrame,
    calibration_sensors: list[str],
    deployment_sensors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks aligned with metadata rows for Sub-steps B–D."""
    sensor_col = metadata["sensor_id"].astype(str)
    cal_set = set(calibration_sensors)
    dep_set = set(deployment_sensors)
    cal_mask = sensor_col.isin(cal_set).to_numpy()
    dep_mask = sensor_col.isin(dep_set).to_numpy()
    return cal_mask, dep_mask


def inventory_to_summary_table(inventory: dict[str, Any]) -> pd.DataFrame:
    """Flat per-sensor table for CSV/HTML export."""
    rows: list[dict[str, Any]] = []
    for sid in inventory["sensor_ids"]:
        s = inventory["sensors"][sid]
        rows.append(
            {
                "sensor_id": sid,
                "n_segments": s["n_segments"],
                "duration_hours": s["duration_hours"],
                "n_global_classes": len(s["global_labels"]),
                "n_runs": len(s["run_ids"]),
                "global_labels": ", ".join(s["global_labels"]),
                "intermediate_labels": ", ".join(s["intermediate_labels"]),
                "min_segments_per_class": s["min_segments_per_global_class"],
            }
        )
    return pd.DataFrame(rows)


def write_inventory_json(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_inventory_html(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    """Minimal styled HTML table for teammate review."""
    headers = list(df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in headers)
        rows_html.append(f"<tr>{cells}</tr>")

    thead = "".join(f"<th>{h}</th>" for h in headers)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 2rem; background: #f8fafc; color: #1a202c; }}
    h1 {{ color: #1a365d; font-size: 1.75rem; margin-bottom: 0.5rem; }}
    p {{ color: #4a5568; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; background: white;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); border-radius: 8px; overflow: hidden; }}
    th {{ background: #2c5282; color: white; text-align: left; padding: 0.75rem 1rem; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #e2e8f0; }}
    tr:nth-child(even) td {{ background: #f7fafc; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Scene proxy: sensor_id (M3N-VC deployment node). Used for calibration-maintenance WIP.</p>
  <table>
    <thead><tr>{thead}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</body>
</html>"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
