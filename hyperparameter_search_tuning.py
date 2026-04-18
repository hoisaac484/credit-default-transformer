"""
Random search over transformer hyperparameters.
Samples n_trials configurations uniformly at random from the search space
and saves results to outputs/hparam_search.json.

Random search is preferred over grid search for high-dimensional spaces
(Bergstra & Bengio, 2012): it explores more of the space in fewer trials
because not every parameter matters equally.

Usage:
    python hyperparameter_search_tuning.py                  # 30 trials (default)
    python hyperparameter_search_tuning.py --n_trials 40    # more trials if time allows
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from data_loader import get_dataloaders
from transformer_model import TemporalTransformer
from train_transformer import train_epoch, evaluate


# ── Search space ───────────────────────────────────────────────────────────────
# Each key maps to a list of candidate values sampled uniformly at random.
# Rationale:
#   d_model    : 64 (baseline) vs 128 (more capacity, ~4x params)
#   n_heads    : must divide d_model; invalid combos filtered below
#   n_layers   : baseline 2 may underfit; 3 adds depth
#   d_ff       : standard 2-4x d_model ratio
#   dropout    : light (0.1) vs moderate (0.2, 0.3) regularisation
#   lr         : bracket the baseline 1e-3 in both directions
#   warmup     : short (3) vs standard (5) linear warmup epochs
SEARCH_SPACE = {
    "d_model"      : [64, 128],
    "n_heads"      : [2, 4, 8],
    "n_layers"     : [2, 3],
    "d_ff"         : [128, 256],
    "dropout"      : [0.1, 0.2, 0.3],
    "lr"           : [2e-3, 1e-3, 5e-4],
    "warmup_epochs": [3, 5],
}


def sample_config(rng: random.Random) -> dict:
    """Sample one valid configuration (n_heads must divide d_model)."""
    while True:
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        if cfg["d_model"] % cfg["n_heads"] == 0:
            return cfg


def run_sweep(n_trials: int = 30, max_epochs: int = 25,
              patience: int = 5, seed: int = 42):

    rng    = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Trials : {n_trials}  |  max_epochs: {max_epochs}  |  patience: {patience}\n")

    train_loader, val_loader, _, pos_weight, _ = get_dataloaders(
        batch_size=256, seed=seed
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    records = []
    t_start = time.time()

    for i in range(n_trials):
        hp = sample_config(rng)
        print(f"[{i+1:2d}/{n_trials}] {hp}")

        torch.manual_seed(seed)
        model = TemporalTransformer(
            d_model  = hp["d_model"],
            n_heads  = hp["n_heads"],
            n_layers = hp["n_layers"],
            d_ff     = hp["d_ff"],
            dropout  = hp["dropout"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hp["lr"], weight_decay=1e-4
        )

        # Cosine LR with linear warmup — mirrors train_transformer.py
        def make_lr_lambda(warmup, total):
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_lambda

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(hp["warmup_epochs"], max_epochs)
        )

        best_auroc     = 0.0
        patience_count = 0

        for epoch in range(1, max_epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, _, val_auroc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_auroc > best_auroc:
                best_auroc     = val_auroc
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                break

        elapsed   = (time.time() - t_start) / 60
        remaining = (elapsed / (i + 1)) * (n_trials - i - 1)
        print(f"         -> val AUROC: {best_auroc:.4f}  "
              f"stopped @ epoch {epoch}  params: {n_params:,}  "
              f"elapsed: {elapsed:.1f} min  est. remaining: {remaining:.1f} min")

        records.append({
            **hp,
            "n_params"      : n_params,
            "best_val_auroc": best_auroc,
            "stopped_epoch" : epoch,
        })

    # Sort by validation AUROC
    records.sort(key=lambda r: r["best_val_auroc"], reverse=True)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/hparam_search.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n=== Top-5 configurations ===")
    for r in records[:5]:
        print(f"  AUROC {r['best_val_auroc']:.4f} | "
              f"d_model={r['d_model']} n_heads={r['n_heads']} "
              f"n_layers={r['n_layers']} d_ff={r['d_ff']} "
              f"dropout={r['dropout']} lr={r['lr']} "
              f"warmup={r['warmup_epochs']} | "
              f"params={r['n_params']:,}")

    best = records[0]
    print(f"\n── Best config found ──")
    print(f"  Run final training with:")
    print(f"  python train_transformer.py "
          f"--d_model {best['d_model']} "
          f"--n_heads {best['n_heads']} "
          f"--n_layers {best['n_layers']} "
          f"--d_ff {best['d_ff']} "
          f"--dropout {best['dropout']} "
          f"--lr {best['lr']} "
          f"--warmup_epochs {best['warmup_epochs']} "
          f"--epochs 60 --patience 12")

    print(f"\nFull results saved to {out_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random hyperparameter search")
    parser.add_argument("--n_trials",   type=int, default=30,
                        help="Number of random configs to try (default: 30)")
    parser.add_argument("--max_epochs", type=int, default=25,
                        help="Max epochs per trial (default: 25)")
    parser.add_argument("--patience",   type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_sweep(
        n_trials   = args.n_trials,
        max_epochs = args.max_epochs,
        patience   = args.patience,
        seed       = args.seed,
    )