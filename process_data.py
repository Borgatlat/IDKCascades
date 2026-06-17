# ASSISTANCE FROM CODEX

"""Utilities for processing the original M3N-VC h24 subset."""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_H24_DIR = Path("datasets/h24")
DEFAULT_OUTPUT_DIR = Path("datasets/processed")


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


def _read_and_segment_file(
    file_path: Path,
    suffix: str,
    segment_seconds: float,
    timestamp_col: str,
) -> pd.DataFrame:
    df = pd.read_parquet(file_path).copy()

    if timestamp_col not in df.columns:
        raise ValueError(f"{file_path} does not contain a '{timestamp_col}' column.")

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

    mic_segments = pd.concat(
        [
            _read_and_segment_file(fp, "_mic", segment_seconds, timestamp_col)
            for fp in mic_files
        ],
        ignore_index=True,
    )
    geo_segments = pd.concat(
        [
            _read_and_segment_file(fp, "_geo", segment_seconds, timestamp_col)
            for fp in geo_files
        ],
        ignore_index=True,
    )

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


def save_h24_spectrogram_arrays(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_H24_DIR,
    segment_seconds: float = 2.0,
    n_fft: int = 256,
    hop_length: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Create and save spectrogram/label arrays for h24 mic and geo data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mic_df, geo_df = load_h24_two_second_segments(
        data_dir=data_dir,
        segment_seconds=segment_seconds,
    )

    mic_spectrograms, mic_labels = segments_to_spectrograms(
        mic_df,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    geo_spectrograms, geo_labels = segments_to_spectrograms(
        geo_df,
        n_fft=n_fft,
        hop_length=hop_length,
    )

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
