"""CLI for hierarchical Ki training (K0–K6) on h24 with loss benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_data import save_h24_paired_arrays
from training.profile import profile_ki_wcet
from training.trainer import benchmark_losses, train_ki
from utils.labels import KI_REGISTRY

DEFAULT_H24_DIR = Path("datasets/h24/h24")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
ALL_KI = tuple(KI_REGISTRY.keys())


def load_best_loss(checkpoint_dir: Path, ki_name: str) -> str:
    path = checkpoint_dir / f"loss_benchmark_{ki_name}.json"
    if not path.exists():
        return "weighted_ce"
    data = json.loads(path.read_text())
    return data.get("best_loss", "weighted_ce")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical Ki on h24")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--benchmark-losses", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--ki", choices=[*ALL_KI, "all"], default="K2")
    parser.add_argument("--loss", default=None, help="Force loss; else use benchmark winner")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_H24_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--benchmark-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--no-augment", action="store_true", help="Disable SpecAugment on train")
    parser.add_argument("--profile-wcet", action="store_true", help="Profile WCET only (no training)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any([args.preprocess, args.benchmark_losses, args.train, args.train_all, args.profile_wcet]):
        print("Use --preprocess, --benchmark-losses, --train, --train-all, and/or --profile-wcet")
        return

    if args.preprocess:
        print(f"Preprocessing paired h24: {args.data_dir}")
        mic, geo, meta = save_h24_paired_arrays(
            output_dir=args.processed_dir,
            data_dir=args.data_dir,
        )
        print(f"Paired mic {mic.shape}, geo {geo.shape}, metadata rows {len(meta)}")

    targets = list(ALL_KI) if args.ki == "all" or args.train_all else [args.ki]

    if args.profile_wcet:
        report = []
        for ki in targets:
            timing = profile_ki_wcet(
                ki_name=ki,
                processed_dir=args.processed_dir,
                batch_size=1,
            )
            report.append(timing)
            print(
                f"{ki}: avg={timing['avg_ms']:.2f}ms WCET={timing['wcet_ms']:.2f}ms "
                f"p95={timing['p95_ms']:.2f}ms"
            )
        out = args.checkpoint_dir / "wcet_profile.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out}")
        return

    augment_train = not args.no_augment

    if args.benchmark_losses:
        for ki in targets:
            benchmark_losses(
                ki_name=ki,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.benchmark_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )

    if args.train or args.train_all:
        summary = []
        for ki in targets:
            loss_key = args.loss or load_best_loss(args.checkpoint_dir, ki)
            print(f"\n>>> Full train {ki} with loss={loss_key}")
            result = train_ki(
                ki_name=ki,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                loss_key=loss_key,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                warmup_epochs=args.warmup_epochs,
                augment_train=augment_train,
            )
            summary.append(
                {
                    "ki": result.ki,
                    "loss": result.loss_key,
                    "val_macro_f1": result.best_val_macro_f1,
                    "val_loss": result.best_val_loss,
                    "params": result.num_params,
                    "inference_avg_ms": result.inference_avg_ms,
                    "inference_wcet_ms": result.inference_wcet_ms,
                    "checkpoint": result.checkpoint,
                }
            )
        out = args.checkpoint_dir / "training_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
