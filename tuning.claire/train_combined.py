"""
Retrain the best transformer config on train+val combined (90% of data),
then evaluate on the held-out test set.

Why: the hyperparameter search already consumed the val set for model selection,
so retraining on train+val gives the final model more data without leaking into test.

Best config (rank-5 from hyperparameter_search_tuning.py):
    d_model=64, n_heads=8, n_layers=3, d_ff=128, dropout=0.2,
    lr=0.001, wd=0.0005, batch_size=128, label_smoothing=0.1, warmup_epochs=3

Fixed epochs = 17 (epoch at which val AUROC peaked in the val-monitored run).

Usage:
    python train_combined.py
"""

import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_fn
from torch.utils.data import DataLoader

from data_loader import get_dataloaders, CreditCardDataset
from train_transformer import train_epoch, evaluate
from transformer_model import TemporalTransformer

# ── Best config ────────────────────────────────────────────────────────────────
HP = dict(
    d_model       = 64,
    n_heads       = 8,
    n_layers      = 3,
    d_ff          = 128,
    dropout       = 0.2,
    lr            = 1e-3,
    weight_decay  = 5e-4,
    batch_size    = 128,
    label_smoothing = 0.1,
    warmup_epochs = 3,
    fixed_epochs  = 17,   # best epoch from val-monitored run
    seed          = 42,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(HP["seed"])

    # ── Load data and combine train + val ──────────────────────────────────────
    _, _, test_loader, pos_weight, splits = get_dataloaders(
        batch_size=HP["batch_size"], seed=HP["seed"]
    )
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits

    X_combined = pd.concat([X_tr, X_val]).reset_index(drop=True)
    y_combined = pd.concat([y_tr, y_val]).reset_index(drop=True)

    combined_ds = CreditCardDataset(X_combined, y_combined, fit_scalers=True)
    train_loader = DataLoader(combined_ds, batch_size=HP["batch_size"], shuffle=True)

    n_train = len(combined_ds)
    print(f"Training on {n_train} samples (train+val combined), testing on {len(X_te)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TemporalTransformer(
        d_model  = HP["d_model"],
        n_heads  = HP["n_heads"],
        n_layers = HP["n_layers"],
        d_ff     = HP["d_ff"],
        dropout  = HP["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )

    total_epochs = HP["fixed_epochs"]

    def lr_lambda(epoch):
        if epoch < HP["warmup_epochs"]:
            return (epoch + 1) / HP["warmup_epochs"]
        progress = (epoch - HP["warmup_epochs"]) / max(1, total_epochs - HP["warmup_epochs"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Train for fixed epochs (no early stopping) ─────────────────────────────
    print(f"\nTraining for {total_epochs} fixed epochs (no early stopping)")
    print(f"{'Epoch':>5}  {'Tr Loss':>8}")
    print("-" * 20)

    for epoch in range(1, total_epochs + 1):
        tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device,
                                  label_smoothing=HP["label_smoothing"])
        scheduler.step()
        print(f"{epoch:5d}  {tr_loss:8.4f}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    test_loss, test_acc, test_auroc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}  AUROC={test_auroc:.4f}")

    # Threshold tuning on combined training set
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for demo, monthly, labels in train_loader:
            logits, _ = model(demo.to(device), monthly.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    tr_probs  = torch.sigmoid(torch.cat(all_logits)).numpy()
    tr_labels = torch.cat(all_labels).numpy().astype(int)

    best_f1, best_thresh = 0.0, 0.5
    for t in [x / 100 for x in range(20, 70)]:
        f1 = f1_fn(tr_labels, (tr_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"Best threshold (train F1={best_f1:.4f}): {best_thresh:.2f}")

    # Detailed test report
    all_logits, all_labels = [], []
    with torch.no_grad():
        for demo, monthly, labels in test_loader:
            logits, _ = model(demo.to(device), monthly.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    probs     = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)

    tuned_preds    = (probs >= best_thresh).astype(int)
    tuned_test_acc = accuracy_score(labels_np, tuned_preds)

    for thresh, label in [(0.5, "@ threshold 0.50"),
                          (best_thresh, f"@ threshold {best_thresh:.2f} (tuned)")]:
        preds = (probs >= thresh).astype(int)
        print(f"\nClassification Report (test {label}):")
        print(classification_report(labels_np, preds, target_names=["No Default", "Default"]))
        print("Confusion Matrix:")
        print(confusion_matrix(labels_np, preds))

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/transformer_best_combined.pt")

    results = {
        "test_loss"      : test_loss,
        "test_acc"       : test_acc,
        "test_acc_tuned" : tuned_test_acc,
        "test_auroc"     : test_auroc,
        "best_threshold" : best_thresh,
        "n_params"       : n_params,
        "n_train"        : n_train,
        "hp"             : HP,
    }
    with open("outputs/transformer_results_combined.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nArtefacts saved to outputs/")
    return results


if __name__ == "__main__":
    main()
