"""Train position-specific RF and MLP skip models.

One model is trained per cascade transition position (e.g. "initial_K0",
"spec_K0_suv_K4"). Each model is tiny and trains in seconds. At runtime only
one model is ever called per IDK transition -- never all of them at once.

RF:  50 trees, depth 4, balanced class weights. Trains in ~1-3 seconds per
     position. Inference ~0.05-0.15ms per sample.
MLP: 2-layer network, hidden=32, input=3 or 7 features. Trains in ~5-15
     seconds per position (20 epochs). Inference ~0.005-0.01ms per sample.

Usage
-----
    # 1. Build datasets (once, after running empirical_outcomes.py)
    python -m cascade.skipper_dataset

    # 2. Train RF skippers for all positions
    python -m cascade.train_skippers --mode rf

    # 3. Train MLP skippers for all positions
    python -m cascade.train_skippers --mode mlp

    # 4. Run the cascade with skip decisions
    python -m cascade.runtime_cascade --mode rf   # or --mode mlp
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from cascade.skipper_dataset import (
    DATASET_DIR,
    PositionDataset,
    load_all_position_datasets,
)

RF_SKIPPER_DIR = Path("checkpoints/skippers/rf")
MLP_SKIPPER_DIR = Path("checkpoints/skippers/mlp")


# ---------------------------------------------------------------------------
# RF
# ---------------------------------------------------------------------------

def train_rf_skipper_for_position(
    ds: PositionDataset,
    n_estimators: int = 50,
    max_depth: int = 4,
    min_samples_leaf: int = 40,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train one RF for one cascade position. Returns the saved payload dict."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X, y = ds.X, ds.y

    # If only 1 class present (e.g. tiny dataset where everyone continues),
    # skip training -- the model would be trivially predicting one class.
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"    [{ds.position_id}] only 1 class in training data -- skipping")
        return {}

    stratify = y if all(np.bincount(y) >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state, stratify=stratify
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0,
                                   target_names=ds.label_meanings[:len(unique_classes)])
    accuracy = float((y_pred == y_test).mean())

    payload = {
        "model": rf,
        "position_id": ds.position_id,
        "chain_type": ds.chain_type,
        "idk_classifier": ds.idk_classifier,
        "remaining": ds.remaining,
        "n_choices": ds.n_choices,
        "is_binary": ds.is_binary,
        "feature_names": ds.feature_names,
        "label_meanings": ds.label_meanings,
        "accuracy": accuracy,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(f"    [{ds.position_id}] RF accuracy={accuracy:.3f}  "
          f"n_train={len(X_train)}  n_choices={ds.n_choices}")
    print("    " + report.replace("\n", "\n    "))
    return payload


def train_all_rf_skippers(
    dataset_dir: str | Path = DATASET_DIR,
    output_dir: str | Path = RF_SKIPPER_DIR,
    **kwargs,
) -> dict[str, dict]:
    """Train one RF per position, save to output_dir/{position_id}.pkl."""
    datasets = load_all_position_datasets(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.perf_counter()
    for pos_id, ds in datasets.items():
        print(f"\n--- {pos_id} ---")
        payload = train_rf_skipper_for_position(ds, **kwargs)
        if payload:
            out = output_dir / f"{pos_id}.pkl"
            with open(out, "wb") as f:
                pickle.dump(payload, f)
            results[pos_id] = payload

    elapsed = time.perf_counter() - t0
    print(f"\nTrained {len(results)} RF skippers in {elapsed:.1f}s  -> {output_dir}")
    return results


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def _class_weights_tensor(y: np.ndarray, n_classes: int):
    import torch
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_mlp_skipper_for_position(
    ds: PositionDataset,
    hidden_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    batch_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    X_raw, y = ds.X.astype(np.float32), ds.y.astype(np.int64)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"    [{ds.position_id}] only 1 class -- skipping")
        return {}

    # Standardize features before training. Without this, mismatched scales
    # (e.g. confidence in [0,1] next to group_size as a raw integer like 2-4,
    # or entropy which can range 0-3+) cause gradient updates to be dominated
    # by whichever feature happens to have the largest raw magnitude -- this
    # was the actual cause of spec_K0_suv_K3 scoring 5.9% accuracy (worse
    # than random on a binary task): the network never learned a useful
    # decision boundary because group_size dwarfed the softmax-derived
    # features in the loss gradient. RF is scale-invariant (tree splits don't
    # care about units) so it never showed this problem.
    feature_mean = X_raw.mean(axis=0)
    feature_std = X_raw.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)  # avoid div-by-zero
    X = (X_raw - feature_mean) / feature_std

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    full_ds = TensorDataset(X_t, y_t)

    train_size = int((1 - test_fraction) * len(full_ds))
    test_size = len(full_ds) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    output_size = ds.n_choices

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size),
    )

    weights = _class_weights_tensor(y, output_size)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0.0

    print(f"    [{ds.position_id}] MLP accuracy={accuracy:.3f}  "
          f"n_train={train_size}  n_choices={ds.n_choices}  "
          f"input_size={input_size}  hidden={hidden_size}")

    payload = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dropout": dropout,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "position_id": ds.position_id,
        "chain_type": ds.chain_type,
        "idk_classifier": ds.idk_classifier,
        "remaining": ds.remaining,
        "n_choices": ds.n_choices,
        "is_binary": ds.is_binary,
        "feature_names": ds.feature_names,
        "label_meanings": ds.label_meanings,
        "accuracy": accuracy,
        "n_train": train_size,
        "n_test": test_size,
    }
    return payload


def train_all_mlp_skippers(
    dataset_dir: str | Path = DATASET_DIR,
    output_dir: str | Path = MLP_SKIPPER_DIR,
    **kwargs,
) -> dict[str, dict]:
    import torch
    datasets = load_all_position_datasets(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.perf_counter()
    for pos_id, ds in datasets.items():
        print(f"\n--- {pos_id} ---")
        payload = train_mlp_skipper_for_position(ds, **kwargs)
        if payload:
            out = output_dir / f"{pos_id}.pt"
            torch.save(payload, out)
            results[pos_id] = payload

    elapsed = time.perf_counter() - t0
    print(f"\nTrained {len(results)} MLP skippers in {elapsed:.1f}s  -> {output_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train position-specific RF or MLP skip models")
    parser.add_argument("--mode", choices=["rf", "mlp", "both"], default="both")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--rf-dir", default=str(RF_SKIPPER_DIR))
    parser.add_argument("--mlp-dir", default=str(MLP_SKIPPER_DIR))
    args = parser.parse_args()

    if args.mode in ("rf", "both"):
        print("\n=== Training RF skippers ===")
        train_all_rf_skippers(args.dataset_dir, args.rf_dir)

    if args.mode in ("mlp", "both"):
        print("\n=== Training MLP skippers ===")
        train_all_mlp_skippers(args.dataset_dir, args.mlp_dir)
