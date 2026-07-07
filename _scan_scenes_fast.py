"""Fast scene inventory scan (filename + timestamp min/max only)."""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from process_data import _file_metadata
from utils.labels import metadata_row_labels

SCENES = ["h08", "h24", "s31", "a06", "i29", "i22"]
SEG = 2.0

SCENE_META = {
    "h08": {"terrain": "Asphalt & gravel", "weather": "Sunny", "targets": "C,G,M,X", "nodes": 6, "paper_h": 2.77},
    "h24": {"terrain": "Asphalt & gravel", "weather": "Rainy", "targets": "C,G,M,X", "nodes": 6, "paper_h": 3.43},
    "s31": {"terrain": "Dirt & gravel", "weather": "Sunny", "targets": "C,G,M,X", "nodes": 6, "paper_h": 2.78},
    "a06": {"terrain": "Asphalt", "weather": "Sunny", "targets": "C,G,X", "nodes": 6, "paper_h": 2.14},
    "i29": {"terrain": "Concrete", "weather": "Windy", "targets": "C,G,M,X", "nodes": 8, "paper_h": 4.14},
    "i22": {"terrain": "Concrete", "weather": "Sunny", "targets": "C,M,X", "nodes": 8, "paper_h": 3.00},
}


def find_raw_dir(scene: str) -> Path:
    base = Path("datasets") / scene
    hits = sorted(base.rglob("*_mic.parquet"))
    if not hits:
        raise FileNotFoundError(scene)
    parents = {p.parent for p in hits}
    return sorted(parents)[0]


def segment_count(file_path: Path) -> tuple[int, str]:
    """Return (n_segments, status) for one mic parquet file."""
    pf = pq.ParquetFile(file_path)
    rows = pf.metadata.num_rows
    if rows == 0:
        return 0, "empty"

    t0: float | None = None
    t1: float | None = None
    scale = 1.0
    for batch in pf.iter_batches(batch_size=500_000, columns=["timestamp"]):
        col = batch.column(0)
        if t0 is None:
            first = col[0].as_py()
            scale = 0.001 if first > 1e11 else 1.0
        b0 = col[0].as_py() * scale
        b1 = col[-1].as_py() * scale
        t0 = b0 if t0 is None else min(t0, b0)
        t1 = b1 if t1 is None else max(t1, b1)

    assert t0 is not None and t1 is not None
    return int((t1 - t0) // SEG) + 1, "ok"


def scan_scene(scene: str) -> dict:
    raw = find_raw_dir(scene)
    files = sorted(raw.glob("*_mic.parquet"))
    runs = sorted({_file_metadata(f, "_mic")["run_id"] for f in files})
    labels = sorted({metadata_row_labels(r)["global_label"] for r in runs})

    per_sensor: dict[str, int] = {}
    empty_files: list[str] = []
    total = 0
    for fp in files:
        meta = _file_metadata(fp, "_mic")
        n, status = segment_count(fp)
        if status == "empty":
            empty_files.append(fp.name)
        per_sensor[meta["sensor_id"]] = per_sensor.get(meta["sensor_id"], 0) + n
        total += n

    active_sensors = {k: v for k, v in per_sensor.items() if v > 0}
    hours_per_sensor = [v * SEG / 3600.0 for v in active_sensors.values()]

    return {
        "scene": scene,
        "raw_dir": str(raw),
        "meta": SCENE_META[scene],
        "n_files": len(files),
        "n_empty": len(empty_files),
        "empty_files": empty_files,
        "runs": runs,
        "labels": labels,
        "sensors": sorted(per_sensor.keys()),
        "n_sensors_active": len(active_sensors),
        "segments_total": total,
        "hours_total": round(total * SEG / 3600.0, 2),
        "hours_per_sensor": {k: round(v * SEG / 3600.0, 2) for k, v in sorted(per_sensor.items())},
        "hours_min": round(min(hours_per_sensor), 2) if hours_per_sensor else 0.0,
        "hours_max": round(max(hours_per_sensor), 2) if hours_per_sensor else 0.0,
        "hours_mean": round(sum(hours_per_sensor) / len(hours_per_sensor), 2) if hours_per_sensor else 0.0,
    }


def main() -> None:
    for scene in SCENES:
        info = scan_scene(scene)
        m = info["meta"]
        print(f"\n{'='*70}")
        print(f"{info['scene'].upper()} | {m['terrain']} | {m['weather']}")
        print(f"Paper: targets={m['targets']} nodes={m['nodes']} length={m['paper_h']}h")
        print(f"Raw: {info['raw_dir']} ({info['n_files']} mic files, {info['n_empty']} empty)")
        print(f"Runs ({len(info['runs'])}): {', '.join(info['runs'])}")
        print(f"Vehicle labels (from run map): {', '.join(info['labels'])}")
        print(f"Sensors ({info['n_sensors_active']}/{len(info['sensors'])} active): {', '.join(info['sensors'])}")
        print(f"Measured: {info['segments_total']:,} segments = {info['hours_total']:.2f} h total")
        print(f"Per-sensor hours: min={info['hours_min']:.2f} mean={info['hours_mean']:.2f} max={info['hours_max']:.2f} (paper={m['paper_h']:.2f})")
        for sid, hrs in info["hours_per_sensor"].items():
            mark = "OK" if hrs > 0.5 else "EMPTY/LOW"
            print(f"  {sid}: {hrs:.2f} h [{mark}]")
        if info["empty_files"]:
            print(f"Empty files ({len(info['empty_files'])}): {', '.join(info['empty_files'][:8])}{'...' if len(info['empty_files'])>8 else ''}")


if __name__ == "__main__":
    main()
