"""Preprocess M3N-VC scene folders into paired spectrogram arrays.

Examples:
  python process_scenes.py --scan
  python process_scenes.py --scenes h08,i29
  python process_scenes.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_data import KNOWN_SCENES, save_scene_paired_arrays

from _scan_scenes_fast import SCENE_META, scan_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and preprocess M3N-VC scenes")
    parser.add_argument("--scan", action="store_true", help="Write checkpoints/scene_catalog.json only")
    parser.add_argument("--scenes", type=str, default="", help="Comma-separated scene ids (e.g. h08,i29)")
    parser.add_argument("--all", action="store_true", help=f"Process all known scenes: {', '.join(KNOWN_SCENES)}")
    parser.add_argument("--out-json", type=Path, default=Path("checkpoints/scene_catalog.json"))
    args = parser.parse_args()

    scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.all:
        scene_ids = list(KNOWN_SCENES)
    if args.scan or not scene_ids:
        catalog = {sid: scan_scene(sid) for sid in KNOWN_SCENES}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}")
        if args.scan and not args.all and not args.scenes:
            return

    for scene_id in scene_ids:
        meta = SCENE_META.get(scene_id, {})
        print(f"\n=== Preprocessing {scene_id} ({meta.get('terrain','?')}, {meta.get('weather','?')}) ===")
        mic, geo, metadata = save_scene_paired_arrays(scene_id)
        out_dir = Path("datasets/processed") / scene_id
        print(
            f"Done {scene_id}: mic={mic.shape} geo={geo.shape} "
            f"segments={len(metadata):,} -> {out_dir}"
        )


if __name__ == "__main__":
    main()
