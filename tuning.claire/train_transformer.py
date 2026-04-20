"""
Training loop for the TemporalTransformer.

Usage:
    python train_transformer.py                     # default hyperparams
    python train_transformer.py --d_model 128 --n_heads 4 --n_layers 2 --epochs 30
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

from data_loader import get_dataloaders
from transformer_model import TemporalTransformer


# ── Training utilities ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, label_smoothing: float = 0.0):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for demo, monthly, labels in loader:
        demo, monthly, labels = demo.to(device), monthly.to(device), labels.to(device)
        if label_smoothing > 0.0:
            labels = labels * (1 - label_smoothing) + (1 - labels) * label_smoothing
        optimizer.zero_grad()
        logits, _ = model(demo, monthly)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels.long()).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_logits, all_labels = [], []

    for demo, monthly, labels in loader:
        demo, monthly, labels = demo.to(device), monthly.to(device), labels.to(device)
        logits, _ = model(demo, monthly)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_n    += labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)
    probs  = torch.sigmoid(logits_all).numpy()
    preds  = (probs >= 0.5).astype(int)
    labels_np = labels_all.numpy().astype(int)

    acc  = accuracy_score(labels_np, preds)
    auroc = roc_auc_score(labels_np, probs)
    avg_loss = total_loss / total_n
    return avg_loss, acc, auroc


# ── Main training function ─────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed PyTorch RNG so weight init and dropout are reproducible
    torch.manual_seed(args.seed)

    # Data
    train_loader, val_loader, test_loader, pos_weight, splits = get_dataloaders(
        batch_size=args.batch_size, seed=args.seed
    )

    # Model
    model = TemporalTransformer(
        d_model  = args.d_model,
        n_heads  = args.n_heads,
        n_layers = args.n_layers,
        d_ff     = args.d_ff,
        dropout  = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Loss: weighted BCE to handle class imbalance (~22% positive)
    pos_weight = pos_weight.to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimiser: AdamW with linear warmup then cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_lambda(epoch):
        """Linear warmup for warmup_epochs, then cosine decay."""
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auroc": []}
    best_auroc = 0.0
    best_state = None
    patience_count = 0

    print(f"\n{'Epoch':>5}  {'Tr Loss':>8}  {'Val Loss':>8}  {'Val Acc':>8}  {'Val AUROC':>9}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                      label_smoothing=args.label_smoothing)
        val_loss, val_acc, val_auroc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        print(f"{epoch:5d}  {tr_loss:8.4f}  {val_loss:8.4f}  {val_acc:8.4f}  {val_auroc:9.4f}"
              f"  ({time.time()-t0:.1f}s)")

        if val_auroc > best_auroc:
            best_auroc  = val_auroc
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Final test evaluation
    test_loss, test_acc, test_auroc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}  AUROC={test_auroc:.4f}")

    # ── Threshold tuning on the TRAINING set ─────────────────────────────────
    # We use the training set (not the validation set) to avoid double-dipping:
    # the val set was already used to select the best checkpoint via early stopping,
    # so optimising the threshold on it would inflate reported performance.
    model.eval()
    tr_logits_all, tr_labels_all = [], []
    with torch.no_grad():
        for demo, monthly, labels in train_loader:
            logits, _ = model(demo.to(device), monthly.to(device))
            tr_logits_all.append(logits.cpu())
            tr_labels_all.append(labels)
    tr_probs  = torch.sigmoid(torch.cat(tr_logits_all)).numpy()
    tr_labels = torch.cat(tr_labels_all).numpy().astype(int)

    from sklearn.metrics import f1_score as f1_fn
    best_f1, best_thresh = 0.0, 0.5
    for t in [x / 100 for x in range(20, 70)]:
        f1 = f1_fn(tr_labels, (tr_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"Best threshold (train F1={best_f1:.4f}): {best_thresh:.2f}")

    # ── Detailed classification report on test set ────────────────────────────
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

    # Save artefacts
    os.makedirs("outputs", exist_ok=True)
    torch.save(best_state, "outputs/transformer_best.pt")
    with open("outputs/transformer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    results = {
        "test_loss"         : test_loss,
        "test_acc"          : test_acc,           # at threshold 0.50
        "test_acc_tuned"    : tuned_test_acc,     # at threshold tuned on train set
        "test_auroc"        : test_auroc,
        "best_val_auroc"    : best_auroc,
        "best_threshold"    : best_thresh,
        "n_params"          : n_params,
        "args"              : vars(args),
    }
    with open("outputs/transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nArtefacts saved to outputs/")
    return model, history, results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Temporal Transformer")
    p.add_argument("--d_model",    type=int,   default=64)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--d_ff",       type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-4,  help="weight decay")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--patience",      type=int,   default=10,  help="early stopping")
    p.add_argument("--warmup_epochs", type=int,   default=5,   help="LR warmup epochs")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing factor (0=off, 0.05 recommended)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
