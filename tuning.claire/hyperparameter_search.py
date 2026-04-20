"""
Grid search over key transformer hyperparameters.
Runs a quick sweep and saves results to outputs/hparam_search.json.

Usage:
    python hyperparameter_search.py
"""

import json
import os
import itertools

import torch
import torch.nn as nn

from data_loader import get_dataloaders
from transformer_model import TemporalTransformer
from train_transformer import train_epoch, evaluate


def run_sweep(max_epochs=20, patience=5, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, pos_weight, _ = get_dataloaders(
        batch_size=256, seed=seed
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Hyperparameter grid
    grid = {
        "d_model" : [32, 64],
        "n_heads" : [2, 4],
        "n_layers": [1, 2],
        "d_ff"    : [64, 128],
        "dropout" : [0.1, 0.2],
        "lr"      : [1e-3, 5e-4],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Total configurations: {len(combos)}")

    records = []

    for i, values in enumerate(combos):
        hp = dict(zip(keys, values))
        print(f"\n[{i+1}/{len(combos)}] {hp}")

        model = TemporalTransformer(
            d_model=hp["d_model"], n_heads=hp["n_heads"],
            n_layers=hp["n_layers"], d_ff=hp["d_ff"],
            dropout=hp["dropout"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hp["lr"], weight_decay=1e-4
        )

        best_auroc     = 0.0
        patience_count = 0

        for epoch in range(1, max_epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, val_acc, val_auroc = evaluate(model, val_loader, criterion, device)

            if val_auroc > best_auroc:
                best_auroc     = val_auroc
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

        record = {**hp, "best_val_auroc": best_auroc, "stopped_epoch": epoch}
        records.append(record)
        print(f"  → best val AUROC: {best_auroc:.4f}  (epoch {epoch})")

    # Sort by best AUROC
    records.sort(key=lambda r: r["best_val_auroc"], reverse=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hparam_search.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\n=== Top-5 configurations ===")
    for r in records[:5]:
        print(r)

    print(f"\nFull results saved to outputs/hparam_search.json")
    return records


if __name__ == "__main__":
    run_sweep()
