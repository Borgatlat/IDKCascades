"""Export cascade classifier table to a flat CSV for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.classifier_registry import ClassifierRegistry

COLUMNS = [
    "name",
    "runtime_ms",
    "p_idk",
    "p_correct",
    "confusion_matrix",
    "allowed_next",
]

NESTED_COLS = ("confusion_matrix", "allowed_next")


def export_registry_csv(
    registry_path: Path,
    output_path: Path,
    *,
    # * makes print_preview keyword-only: export_registry_csv(path, out, print_preview=False)
    print_preview: bool = True,
) -> pd.DataFrame:
    registry = ClassifierRegistry.load(registry_path)
    df = registry.to_dataframe()

    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Registry missing columns: {missing}")

    out = df[COLUMNS].copy()
    out = out.sort_values("name").reset_index(drop=True)

    for col in NESTED_COLS:
        # json.dumps turns nested lists/dicts into one CSV-safe string per cell
        out[col] = out[col].apply(json.dumps)

    for col in ("runtime_ms", "p_idk", "p_correct"):
        out[col] = out[col].round(6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if print_preview:
        print(out.to_string(index=False))
        print(f"\nWrote {output_path} ({len(out)} rows)")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classifier registry to CSV")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("checkpoints/classifier_registry.json"),
        help="Path to classifier_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/classifier_registry_table.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    export_registry_csv(args.registry, args.output)


if __name__ == "__main__":
    main()
