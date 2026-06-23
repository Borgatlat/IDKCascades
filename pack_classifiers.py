"""Build dist/IDKCascade_classifiers.zip for teammate handoff (Teams, email, etc.)."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def build_bundle(repo_root: Path, output_zip: Path) -> None:
    staging = repo_root / "dist" / "_classifiers_staging"
    if staging.exists():
        shutil.rmtree(staging)

    ckpt_dst = staging / "checkpoints"
    ckpt_dst.mkdir(parents=True)

    src_ckpt = repo_root / "checkpoints"
    for name in KI_NAMES:
        shutil.copy2(src_ckpt / f"{name}.pt", ckpt_dst / f"{name}.pt")
        metrics = src_ckpt / f"{name}_metrics.json"
        if metrics.exists():
            shutil.copy2(metrics, ckpt_dst / metrics.name)

    for extra in ("classifier_registry.json", "wcet_profile.json"):
        path = src_ckpt / extra
        if path.exists():
            shutil.copy2(path, ckpt_dst / extra)

    readme = repo_root / "TEAMMATE_SETUP.md"
    if readme.exists():
        shutil.copy2(readme, staging / "TEAMMATE_SETUP.md")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    shutil.rmtree(staging)
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_zip} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack classifier weights for teammate handoff")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/IDKCascade_classifiers.zip"),
    )
    args = parser.parse_args()
    build_bundle(Path(__file__).resolve().parent, args.output)


if __name__ == "__main__":
    main()
