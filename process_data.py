# ASSISTANCE FROM CODEX

"""Utilities for processing the original M3N-VC h24 subset."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


DEFAULT_H24_DIR = Path("datasets/h24/h24")
DEFAULT_OUTPUT_DIR = Path("datasets/processed")
# M3N-VC scenes under datasets/ (each folder may nest scene_id/scene_id/).
KNOWN_SCENES = ("h08", "h24", "s31", "a06", "i29", "i22")


def resolve_scene_raw_dir(scene_id: str, datasets_root: Path | str = "datasets") -> Path:
    """Return the folder containing *_mic.parquet for one M3N-VC scene."""
    base = Path(datasets_root) / scene_id
    if not base.exists():
        raise FileNotFoundError(f"Scene folder not found: {base}")
    if list(base.glob("*_mic.parquet")):
        return base
    nested = base / scene_id
    if nested.exists() and list(nested.glob("*_mic.parquet")):
        return nested
    hits = sorted({p.parent for p in base.rglob("*_mic.parquet")})
    if not hits:
        raise FileNotFoundError(f"No *_mic.parquet files under {base}")
    return hits[0]


def _count_segments_in_file(file_path: Path, segment_seconds: float) -> int:
    """Estimate 2 s segment count from timestamp span (no waveform read)."""
    pf = pq.ParquetFile(file_path)
    if pf.metadata.num_rows == 0:
        return 0
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
    return int((t1 - t0) // segment_seconds) + 1


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    """Convert millisecond Unix timestamps to seconds when detected."""
    if series.empty:
        return series
    values = series.astype("float64")
    if values.max() > 1e11:
        return values / 1000.0
    return values


def _file_metadata(file_path: Path, suffix: str) -> dict[str, str]:
    """Pull run/sensor names from files like run0_rs1_mic.parquet."""
    stem = file_path.stem
    base_name = stem.removesuffix(suffix)
    parts = base_name.split("_")

    return {
        "source_file": file_path.name,
        "run_id": parts[0] if parts else "",
        "sensor_id": "_".join(parts[1:]) if len(parts) > 1 else "",
    }


def _timestamp_bounds(file_path: Path) -> tuple[float, float, float]:
    """Return (t0, t1, scale) for a parquet timestamp column."""
    pf = pq.ParquetFile(file_path)
    if pf.metadata.num_rows == 0:
        raise ValueError(f"{file_path} is empty (0 rows).")
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
    return t0, t1, scale


def _segment_waveforms_from_file(
    file_path: Path,
    segment_seconds: float,
    *,
    sample_col: str = "samples",
    timestamp_col: str = "timestamp",
) -> dict[int, np.ndarray]:
    """Stream one parquet file into per-segment waveforms (low RAM)."""
    first_ts, _, scale = _timestamp_bounds(file_path)
    buffers: dict[int, list[float]] = {}

    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=200_000, columns=[timestamp_col, sample_col]):
        ts_values = batch.column(0).to_numpy(zero_copy_only=True)
        samples = batch.column(1).to_numpy(zero_copy_only=True)
        if scale != 1.0:
            ts_values = ts_values.astype(np.float64, copy=False) * scale
        seg_nums = ((ts_values - first_ts) // segment_seconds).astype(np.int32, copy=False)
        for seg_num, sample in zip(seg_nums, samples, strict=False):
            key = int(seg_num)
            bucket = buffers.get(key)
            if bucket is None:
                bucket = []
                buffers[key] = bucket
            bucket.append(float(sample))

    if not buffers:
        raise ValueError(f"{file_path} produced no segments.")
    return {seg: np.asarray(wave, dtype=np.float32) for seg, wave in buffers.items()}


def _max_segment_sample_count(file_path: Path, segment_seconds: float) -> int:
    """Return longest 2 s segment length (in samples) inside one parquet file."""
    first_ts, _, scale = _timestamp_bounds(file_path)
    counts: dict[int, int] = {}
    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=200_000, columns=["timestamp", "samples"]):
        ts_values = batch.column(0).to_numpy(zero_copy_only=True)
        if scale != 1.0:
            ts_values = ts_values.astype(np.float64, copy=False) * scale
        seg_nums = ((ts_values - first_ts) // segment_seconds).astype(np.int32, copy=False)
        for seg_num in seg_nums:
            key = int(seg_num)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError(f"{file_path} produced no segments.")
    return max(counts.values())


def _scene_target_samples(mic_files: list[Path], segment_seconds: float, n_fft: int) -> int:
    """Use one fixed waveform length per scene so all spectrograms share shape."""
    target = n_fft
    for mic_path in mic_files:
        try:
            target = max(target, _max_segment_sample_count(mic_path, segment_seconds))
        except ValueError:
            continue
    return target


def _spectrogram_shape(target_samples: int, n_fft: int, hop_length: int) -> tuple[int, int]:
    """Analytic STFT output shape for fixed-length segments."""
    if target_samples < n_fft:
        target_samples = n_fft
    time_frames = 1 + (target_samples - n_fft) // hop_length
    return (n_fft // 2 + 1, time_frames)


def _waveforms_to_spectrograms_with_keys(
    waveforms: dict[int, np.ndarray],
    metadata: dict[str, str],
    *,
    n_fft: int = 256,
    hop_length: int | None = None,
    target_samples: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Convert streamed segment waveforms to spectrograms + metadata rows."""
    if hop_length is None:
        hop_length = n_fft // 2

    run_id = metadata["run_id"]
    sensor_id = metadata["sensor_id"]
    source_file = metadata["source_file"]
    if target_samples is None:
        target_samples = max(len(wave) for wave in waveforms.values())
    if target_samples < n_fft:
        target_samples = n_fft

    window = np.hanning(n_fft).astype(np.float32)
    spectrograms: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for seg_num in sorted(waveforms):
        signal = waveforms[seg_num]
        if signal.size < target_samples:
            signal = np.pad(signal, (0, target_samples - signal.size))
        else:
            signal = signal[:target_samples]

        segment_key = f"{run_id}_{sensor_id}_seg{seg_num:05d}"
        spectrograms.append(_stft_magnitude(signal, n_fft, hop_length, window))
        meta_rows.append(
            {
                "segment_key": segment_key,
                "run_id": run_id,
                "sensor_id": sensor_id,
                "segment_number": seg_num,
                "source_file": source_file,
            }
        )

    return np.stack(spectrograms), meta_rows


def _paired_file_to_spectrograms(
    mic_path: Path,
    geo_path: Path,
    segment_seconds: float,
    *,
    n_fft: int = 256,
    hop_length: int | None = None,
    target_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Stream mic/geo parquet pair into aligned spectrogram batches."""
    mic_meta = _file_metadata(mic_path, "_mic")
    geo_meta = _file_metadata(geo_path, "_geo")
    if mic_meta["run_id"] != geo_meta["run_id"] or mic_meta["sensor_id"] != geo_meta["sensor_id"]:
        raise ValueError(f"Mic/geo metadata mismatch: {mic_path.name} vs {geo_path.name}")

    mic_waves = _segment_waveforms_from_file(mic_path, segment_seconds)
    geo_waves = _segment_waveforms_from_file(geo_path, segment_seconds)
    shared_segments = sorted(set(mic_waves) & set(geo_waves))
    if not shared_segments:
        raise ValueError(f"Mic/geo segment mismatch in {mic_path.name}")

    dropped = (set(mic_waves) | set(geo_waves)) - set(shared_segments)
    if dropped:
        print(
            f"  [warn] {mic_path.name}: {len(shared_segments)} shared segments, "
            f"dropped {len(dropped)} unmatched"
        )

    mic_waves = {seg: mic_waves[seg] for seg in shared_segments}
    geo_waves = {seg: geo_waves[seg] for seg in shared_segments}

    mic_specs, meta_rows = _waveforms_to_spectrograms_with_keys(
        mic_waves, mic_meta, n_fft=n_fft, hop_length=hop_length, target_samples=target_samples
    )
    geo_specs, _ = _waveforms_to_spectrograms_with_keys(
        geo_waves, geo_meta, n_fft=n_fft, hop_length=hop_length, target_samples=target_samples
    )
    return mic_specs, geo_specs, meta_rows


def _read_and_segment_file(
    file_path: Path,
    suffix: str,
    segment_seconds: float,
    timestamp_col: str,
) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()
    if df.empty:
        raise ValueError(f"{file_path} is empty (0 rows).")

    if timestamp_col not in df.columns:
        raise ValueError(f"{file_path} does not contain a '{timestamp_col}' column.")

    df[timestamp_col] = _normalize_timestamps(df[timestamp_col])

    metadata = _file_metadata(file_path, suffix)
    for column, value in metadata.items():
        df[column] = value

    first_timestamp = df[timestamp_col].min()
    segment_number = ((df[timestamp_col] - first_timestamp) // segment_seconds).astype(int)

    df["segment_number"] = segment_number
    df["segment_start"] = first_timestamp + (segment_number * segment_seconds)
    df["segment_end"] = df["segment_start"] + segment_seconds
    df["segment_id"] = (
        df["source_file"].str.removesuffix(suffix + ".parquet")
        + "_seg"
        + df["segment_number"].astype(str).str.zfill(5)
    )

    return df


def load_h24_two_second_segments(
    data_dir: str | Path = DEFAULT_H24_DIR,
    segment_seconds: float = 2.0,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load h24 mic/geo parquet files and label samples by 2 second segment.

    Returns:
        A tuple of ``(mic_segments, geo_segments)`` pandas DataFrames.

    Each returned DataFrame contains the original parquet columns plus:
        ``source_file``, ``run_id``, ``sensor_id``, ``segment_number``,
        ``segment_start``, ``segment_end``, and ``segment_id``.
    """
    data_dir = Path(data_dir)
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be greater than 0.")

    mic_files = sorted(data_dir.glob("*_mic.parquet"))
    geo_files = sorted(data_dir.glob("*_geo.parquet"))

    if not mic_files:
        raise FileNotFoundError(f"No *_mic.parquet files found in {data_dir}.")
    if not geo_files:
        raise FileNotFoundError(f"No *_geo.parquet files found in {data_dir}.")

    mic_frames: list[pd.DataFrame] = []
    for fp in mic_files:
        try:
            mic_frames.append(_read_and_segment_file(fp, "_mic", segment_seconds, timestamp_col))
        except ValueError as exc:
            print(f"  [skip] {fp.name}: {exc}")
    if not mic_frames:
        raise FileNotFoundError(f"No non-empty mic parquet files in {data_dir}.")

    geo_frames: list[pd.DataFrame] = []
    for fp in geo_files:
        try:
            geo_frames.append(_read_and_segment_file(fp, "_geo", segment_seconds, timestamp_col))
        except ValueError as exc:
            print(f"  [skip] {fp.name}: {exc}")
    if not geo_frames:
        raise FileNotFoundError(f"No non-empty geo parquet files in {data_dir}.")

    mic_segments = pd.concat(mic_frames, ignore_index=True)
    geo_segments = pd.concat(geo_frames, ignore_index=True)

    return mic_segments, geo_segments


def run_id_to_class(run_id: str) -> int:
    """Map run0/run1 to class 1, run2/run3 to class 2, and so on."""
    run_number = int(str(run_id).removeprefix("run"))
    return (run_number // 2) + 1


def _stft_magnitude(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop_length]
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, n=n_fft, axis=1)).T.astype(np.float32)


def segments_to_spectrograms(
    segments: pd.DataFrame,
    sample_col: str = "samples",
    segment_col: str = "segment_id",
    run_col: str = "run_id",
    n_fft: int = 256,
    hop_length: int | None = None,
    target_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert segmented samples into a 3D spectrogram array and class labels.

    Args:
        segments: DataFrame returned by ``load_h24_two_second_segments``.
        sample_col: Column containing waveform samples.
        segment_col: Column identifying each 2 second segment.
        run_col: Column containing run ids such as ``run0``.
        n_fft: Number of samples per STFT window.
        hop_length: Number of samples between windows. Defaults to ``n_fft // 2``.
        target_samples: Fixed samples per segment. Defaults to the longest segment.

    Returns:
        ``(spectrograms, labels)`` where ``spectrograms`` has shape
        ``(num_segments, frequency_bins, time_frames)`` and ``labels`` contains
        integer classes where run0/run1 -> 1, run2/run3 -> 2, etc.
    """
    if hop_length is None:
        hop_length = n_fft // 2
    if n_fft <= 0:
        raise ValueError("n_fft must be greater than 0.")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than 0.")

    required_cols = {sample_col, segment_col, run_col}
    missing_cols = required_cols - set(segments.columns)
    if missing_cols:
        raise ValueError(f"segments is missing columns: {sorted(missing_cols)}")

    grouped = segments.groupby(segment_col, sort=True)
    if target_samples is None:
        target_samples = int(grouped.size().max())
    if target_samples < n_fft:
        target_samples = n_fft

    window = np.hanning(n_fft).astype(np.float32)
    spectrograms: list[np.ndarray] = []
    labels: list[int] = []

    for _, segment in grouped:
        signal = segment[sample_col].to_numpy(dtype=np.float32)
        if signal.size < target_samples:
            signal = np.pad(signal, (0, target_samples - signal.size))
        else:
            signal = signal[:target_samples]

        spectrograms.append(_stft_magnitude(signal, n_fft, hop_length, window))
        labels.append(run_id_to_class(segment[run_col].iloc[0]))

    return np.stack(spectrograms), np.array(labels, dtype=np.int64)


def segments_to_spectrograms_with_keys(
    segments: pd.DataFrame,
    sample_col: str = "samples",
    segment_col: str = "segment_id",
    run_col: str = "run_id",
    sensor_col: str = "sensor_id",
    segment_num_col: str = "segment_number",
    n_fft: int = 256,
    hop_length: int | None = None,
    target_samples: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Like segments_to_spectrograms but also returns per-segment metadata dicts."""
    if hop_length is None:
        hop_length = n_fft // 2

    required_cols = {sample_col, segment_col, run_col, sensor_col, segment_num_col}
    missing_cols = required_cols - set(segments.columns)
    if missing_cols:
        raise ValueError(f"segments is missing columns: {sorted(missing_cols)}")

    grouped = segments.groupby(segment_col, sort=True)
    if target_samples is None:
        target_samples = int(grouped.size().max())
    if target_samples < n_fft:
        target_samples = n_fft

    window = np.hanning(n_fft).astype(np.float32)
    spectrograms: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for _, segment in grouped:
        signal = segment[sample_col].to_numpy(dtype=np.float32)
        if signal.size < target_samples:
            signal = np.pad(signal, (0, target_samples - signal.size))
        else:
            signal = signal[:target_samples]

        run_id = str(segment[run_col].iloc[0])
        sensor_id = str(segment[sensor_col].iloc[0])
        seg_num = int(segment[segment_num_col].iloc[0])
        segment_key = f"{run_id}_{sensor_id}_seg{seg_num:05d}"

        spectrograms.append(_stft_magnitude(signal, n_fft, hop_length, window))
        meta_rows.append(
            {
                "segment_key": segment_key,
                "run_id": run_id,
                "sensor_id": sensor_id,
                "segment_number": seg_num,
            }
        )

    return np.stack(spectrograms), meta_rows


def _resize_geo_to_mic(geo_spec: np.ndarray, mic_shape: tuple[int, int]) -> np.ndarray:
    """Resize geo STFT to mic grid once at preprocess (avoids interpolate every forward pass)."""
    if geo_spec.shape == mic_shape:
        return geo_spec
    t = torch.from_numpy(geo_spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=mic_shape, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def save_scene_paired_arrays(
    scene_id: str,
    *,
    datasets_root: Path | str = "datasets",
    segment_seconds: float = 2.0,
    n_fft: int = 256,
    hop_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Preprocess one M3N-VC scene into paired mic/geo spectrograms + metadata."""
    data_dir = resolve_scene_raw_dir(scene_id, datasets_root)
    output_dir = Path(datasets_root) / "processed" / scene_id
    prefix = scene_id
    return save_h24_paired_arrays(
        output_dir=output_dir,
        data_dir=data_dir,
        segment_seconds=segment_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
        array_prefix=prefix,
    )


def save_h24_paired_arrays(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_H24_DIR,
    segment_seconds: float = 2.0,
    n_fft: int = 256,
    hop_length: int | None = None,
    array_prefix: str = "h24",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build aligned mic/geo spectrogram pairs and metadata for hierarchical Ki training."""
    from utils.labels import metadata_row_labels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    mic_files = sorted(data_dir.glob("*_mic.parquet"))
    if not mic_files:
        raise FileNotFoundError(f"No *_mic.parquet files found in {data_dir}.")

    if hop_length is None:
        hop_length = n_fft // 2
    target_samples = _scene_target_samples(mic_files, segment_seconds, n_fft)
    spec_shape = _spectrogram_shape(target_samples, n_fft, hop_length)
    print(f"  Scene target_samples={target_samples}, spectrogram shape={spec_shape}")

    paired_mic_path = output_dir / f"{array_prefix}_paired_mic.npy"
    paired_geo_path = output_dir / f"{array_prefix}_paired_geo.npy"
    metadata_rows: list[dict] = []
    write_idx = 0
    mic_mm: np.memmap | None = None
    geo_mm: np.memmap | None = None

    for index, mic_path in enumerate(mic_files, start=1):
        geo_path = mic_path.with_name(mic_path.name.replace("_mic.parquet", "_geo.parquet"))
        if not geo_path.exists():
            raise FileNotFoundError(f"Missing paired geo file for {mic_path.name}")

        print(f"  [{index}/{len(mic_files)}] {mic_path.name}")
        try:
            mic_specs, geo_specs, mic_meta = _paired_file_to_spectrograms(
                mic_path,
                geo_path,
                segment_seconds,
                n_fft=n_fft,
                hop_length=hop_length,
                target_samples=target_samples,
            )
        except ValueError as exc:
            print(f"  [skip] {exc}")
            continue

        if mic_mm is None:
            total_segments = sum(
                _count_segments_in_file(fp, segment_seconds) for fp in mic_files
            )
            mic_mm = np.lib.format.open_memmap(
                paired_mic_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_segments, *spec_shape),
            )
            geo_mm = np.lib.format.open_memmap(
                paired_geo_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_segments, *spec_shape),
            )
            print(f"  Allocated memmap for {total_segments:,} segments {spec_shape}")

        batch_size = len(mic_meta)
        for offset, (row, mic_spec, geo_spec) in enumerate(
            zip(mic_meta, mic_specs, geo_specs)
        ):
            labels = metadata_row_labels(row["run_id"])
            metadata_rows.append({**row, "scene_id": array_prefix, **labels})
            mic_mm[write_idx + offset] = mic_spec
            geo_mm[write_idx + offset] = _resize_geo_to_mic(geo_spec, mic_spec.shape)
        write_idx += batch_size

        del mic_specs, geo_specs
        gc.collect()

    if mic_mm is None or write_idx == 0:
        raise FileNotFoundError(f"No usable mic/geo pairs found in {data_dir}.")

    mic_mm.flush()
    geo_mm.flush()

    if write_idx != mic_mm.shape[0]:
        # Trim oversized memmap (estimate overshoot or skipped files).
        mic_array = np.array(mic_mm[:write_idx], copy=True)
        geo_array = np.array(geo_mm[:write_idx], copy=True)
        del mic_mm, geo_mm
        gc.collect()
        paired_mic_path.unlink(missing_ok=True)
        paired_geo_path.unlink(missing_ok=True)
        np.save(paired_mic_path, mic_array)
        np.save(paired_geo_path, geo_array)
    else:
        mic_array = mic_mm
        geo_array = geo_mm
        del mic_mm, geo_mm
        gc.collect()

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_parquet(output_dir / f"{array_prefix}_metadata.parquet", index=False)

    # Drop stale normalized caches so trainer rebuilds from resized geo.
    for stale in (f"{array_prefix}_paired_mic_norm.npy", f"{array_prefix}_paired_geo_norm.npy"):
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    # Legacy single-modality caches (same data, new layout).
    np.save(output_dir / f"{array_prefix}_mic_spectrograms.npy", np.asarray(mic_array))
    np.save(output_dir / f"{array_prefix}_geo_spectrograms.npy", np.asarray(geo_array))

    return mic_array, geo_array, metadata


def save_h24_spectrogram_arrays(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_H24_DIR,
    segment_seconds: float = 2.0,
    n_fft: int = 256,
    hop_length: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Create and save spectrogram/label arrays for h24 mic and geo data.

    Files are processed one at a time so we do not load the full h24 subset
    (~174M waveform rows) into memory at once.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    mic_files = sorted(data_dir.glob("*_mic.parquet"))
    geo_files = sorted(data_dir.glob("*_geo.parquet"))
    if not mic_files:
        raise FileNotFoundError(f"No *_mic.parquet files found in {data_dir}.")
    if not geo_files:
        raise FileNotFoundError(f"No *_geo.parquet files found in {data_dir}.")

    def _process_files(files: list[Path], suffix: str) -> tuple[np.ndarray, np.ndarray]:
        all_specs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for index, file_path in enumerate(files, start=1):
            print(f"  [{index}/{len(files)}] {file_path.name}")
            segment_df = _read_and_segment_file(
                file_path,
                suffix,
                segment_seconds,
                "timestamp",
            )
            specs, labels = segments_to_spectrograms(
                segment_df,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            all_specs.append(specs)
            all_labels.append(labels)
            del segment_df

        return np.concatenate(all_specs, axis=0), np.concatenate(all_labels, axis=0)

    print("Processing microphone spectrograms...")
    mic_spectrograms, mic_labels = _process_files(mic_files, "_mic")
    print("Processing geophone spectrograms...")
    geo_spectrograms, geo_labels = _process_files(geo_files, "_geo")

    np.save(output_dir / "h24_mic_spectrograms.npy", mic_spectrograms)
    np.save(output_dir / "h24_mic_labels.npy", mic_labels)
    np.save(output_dir / "h24_geo_spectrograms.npy", geo_spectrograms)
    np.save(output_dir / "h24_geo_labels.npy", geo_labels)

    return (mic_spectrograms, mic_labels), (geo_spectrograms, geo_labels)


if __name__ == "__main__":
    (mic_spectrograms, mic_labels), (geo_spectrograms, geo_labels) = (
        save_h24_spectrogram_arrays()
    )
    print(f"Mic spectrograms: {mic_spectrograms.shape}, labels: {mic_labels.shape}")
    print(f"Geo spectrograms: {geo_spectrograms.shape}, labels: {geo_labels.shape}")
