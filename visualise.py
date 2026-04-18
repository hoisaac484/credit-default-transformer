"""
Visualisation utilities for the report:
  1. Attention heatmaps (which tokens attend to which)
  2. Training curves (loss, AUROC) — default and tuned transformer
  3. EDA plots (class balance, correlation, distributions)
  4. Model comparison table/bar chart — THREE models:
       Default Transformer, Tuned Transformer, Random Forest
  5. ROC and Precision-Recall curves — all three models

Run:
    python visualise.py --mode all
    python visualise.py --mode eda
    python visualise.py --mode attention
    python visualise.py --mode curves
    python visualise.py --mode compare
    python visualise.py --mode roc_pr
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from data_loader import load_raw, clean, split_data, get_dataloaders
from transformer_model import TemporalTransformer

os.makedirs("outputs/figures", exist_ok=True)
MONTH_LABELS = ["Sep 05", "Aug 05", "Jul 05", "Jun 05", "May 05", "Apr 05"]
TOKEN_LABELS = ["CLS", "Demo"] + MONTH_LABELS   # 8 tokens


# ── 1. EDA ─────────────────────────────────────────────────────────────────────

def plot_eda():
    X, y = load_raw()
    X    = clean(X)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Class balance
    counts = y.value_counts()
    axes[0].bar(["No Default (0)", "Default (1)"], counts.values,
                color=["#4c9be8", "#e8754c"], edgecolor="white")
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}\n({v/len(y)*100:.1f}%)",
                     ha="center", fontsize=9)

    # Default rate by education level
    edu_map  = {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}
    edu_rate = pd.DataFrame({"EDUCATION": X["EDUCATION"], "default": y})\
                 .groupby("EDUCATION")["default"].mean().rename(index=edu_map)
    axes[1].bar(edu_rate.index, edu_rate.values, color="#7b68ee", edgecolor="white")
    axes[1].set_title("Default Rate by Education")
    axes[1].set_ylabel("Default Rate")
    axes[1].set_ylim(0, 0.35)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Default rate by repayment status (PAY_0 = most recent)
    pay_rate = pd.DataFrame({"PAY_0": X["PAY_0"], "default": y})\
                 .groupby("PAY_0")["default"].mean()
    axes[2].bar(pay_rate.index.astype(str), pay_rate.values,
                color="#50c8a0", edgecolor="white")
    axes[2].set_title("Default Rate by PAY_0 (Sep 2005)")
    axes[2].set_xlabel("Repayment Status")
    axes[2].set_ylabel("Default Rate")
    axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    path = "outputs/figures/eda_overview.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # Correlation heatmap (numeric features)
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, fmt=".1f", linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    path = "outputs/figures/eda_correlation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # Distribution of bill amounts over months
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                 "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, (col, month) in enumerate(zip(bill_cols, MONTH_LABELS)):
        ax = axes[i // 3][i % 3]
        default_vals    = X.loc[y == 1, col] / 1000
        no_default_vals = X.loc[y == 0, col] / 1000
        ax.hist(no_default_vals.clip(-50, 500), bins=40, alpha=0.6,
                color="#4c9be8", label="No Default", density=True)
        ax.hist(default_vals.clip(-50, 500), bins=40, alpha=0.6,
                color="#e8754c", label="Default", density=True)
        ax.set_title(f"Bill Amount — {month}")
        ax.set_xlabel("NT$ (thousands)")
        if i == 0:
            ax.legend(fontsize=8)
    plt.suptitle("Bill Statement Distributions by Month", y=1.01)
    plt.tight_layout()
    path = "outputs/figures/eda_bill_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── 2. Attention heatmaps ──────────────────────────────────────────────────────

def plot_attention(model_path="outputs/transformer_best_tuned.pt",
                   d_model=128, n_heads=8, n_layers=3, d_ff=128, month_dim=5):
    device = torch.device("cpu")

    model = TemporalTransformer(d_model=d_model, n_heads=n_heads,
                                n_layers=n_layers, d_ff=d_ff, month_dim=month_dim)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(batch_size=512)

    # Collect attention weights over test set
    all_attn = [[] for _ in range(n_layers)]

    with torch.no_grad():
        for demo, monthly, _ in test_loader:
            _, attn_maps = model(demo, monthly)
            for layer_idx, attn in enumerate(attn_maps):
                all_attn[layer_idx].append(attn.numpy())

    for layer_idx in range(n_layers):
        attn_avg = np.concatenate(all_attn[layer_idx], axis=0).mean(axis=0)  # (H, 8, 8)
        attn_mean_heads = attn_avg.mean(axis=0)  # (8, 8)

        fig, axes = plt.subplots(1, n_heads + 1, figsize=(4 * (n_heads + 1), 4))

        # Mean across heads
        sns.heatmap(attn_mean_heads, ax=axes[0], cmap="YlOrRd",
                    xticklabels=TOKEN_LABELS, yticklabels=TOKEN_LABELS,
                    annot=True, fmt=".2f", linewidths=0.3, vmin=0)
        axes[0].set_title("Mean (all heads)")
        axes[0].set_xlabel("Key token")
        axes[0].set_ylabel("Query token")

        # Individual heads
        for h in range(n_heads):
            sns.heatmap(attn_avg[h], ax=axes[h + 1], cmap="YlOrRd",
                        xticklabels=TOKEN_LABELS, yticklabels=TOKEN_LABELS,
                        annot=True, fmt=".2f", linewidths=0.3, vmin=0)
            axes[h + 1].set_title(f"Head {h + 1}")
            axes[h + 1].set_xlabel("Key token")
            axes[h + 1].set_ylabel("")

        plt.suptitle(f"Attention Weights — Layer {layer_idx + 1} (averaged over test set)",
                     y=1.02)
        plt.tight_layout()
        path = f"outputs/figures/attention_layer{layer_idx + 1}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

        # CLS token attention
        fig, ax = plt.subplots(figsize=(7, 2.5))
        cls_attn = attn_mean_heads[0]
        ax.bar(TOKEN_LABELS, cls_attn, color="#7b68ee", edgecolor="white")
        ax.set_title(f"Layer {layer_idx + 1}: CLS token attention over all tokens")
        ax.set_ylabel("Attention weight")
        ax.set_ylim(0, cls_attn.max() * 1.25)
        for i, v in enumerate(cls_attn):
            ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        path = f"outputs/figures/cls_attention_layer{layer_idx + 1}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


# ── 3. Training curves — default AND tuned ─────────────────────────────────────

def plot_curves():
    """
    Plots training curves for both default and tuned transformer.
    Saves separate figures and one combined AUROC comparison.
    """
    histories = {}

    for label, path in [
        ("Default",  "outputs/transformer_history_default.json"),
        ("Tuned",    "outputs/transformer_history_tuned.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                histories[label] = json.load(f)
        else:
            print(f"Warning: {path} not found — skipping {label}")

    if not histories:
        print("No transformer history files found.")
        return

    # Individual curves per model
    for label, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_loss"], label="Train loss", color="#4c9be8")
        ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#e8754c")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax1.set_title(f"{label} Transformer — Training & Validation Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, history["val_auroc"], label="Val AUROC", color="#50c8a0")
        ax2.axhline(y=max(history["val_auroc"]), color="grey",
                    linestyle="--", alpha=0.6,
                    label=f"Best: {max(history['val_auroc']):.4f}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUROC")
        ax2.set_title(f"{label} Transformer — Validation AUROC")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"outputs/figures/training_curves_{label.lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

    # Combined AUROC comparison plot
    if len(histories) == 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = {"Default": "#4c9be8", "Tuned": "#e8754c"}
        for label, history in histories.items():
            epochs = range(1, len(history["val_auroc"]) + 1)
            ax.plot(epochs, history["val_auroc"],
                    label=f"{label} (best={max(history['val_auroc']):.4f})",
                    color=colors[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val AUROC")
        ax.set_title("Validation AUROC — Default vs Tuned Transformer")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = "outputs/figures/training_curves_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


# ── 4. ROC and Precision-Recall curves — all three models ─────────────────────

def plot_roc_pr(d_model=128, n_heads=8, n_layers=3, d_ff=128, month_dim=5):
    """Overlay ROC and PR curves for all three models."""
    device = torch.device("cpu")

    _, _, test_loader, _, splits = get_dataloaders(batch_size=512)
    X, y = load_raw()
    X = clean(X)
    _, _, X_te, _, _, y_te = split_data(X, y)
    labels_np = y_te.values.astype(int)

    model_probs = {}

    # Default transformer
    default_path = "outputs/transformer_best_default.pt"
    if os.path.exists(default_path):
        model_def = TemporalTransformer(d_model=64, n_heads=4,
                                        n_layers=2, d_ff=128, month_dim=month_dim)
        state = torch.load(default_path, map_location=device, weights_only=True)
        model_def.load_state_dict(state)
        model_def.eval()
        logits_list = []
        with torch.no_grad():
            for demo, monthly, _ in test_loader:
                logits, _ = model_def(demo, monthly)
                logits_list.append(torch.sigmoid(logits).numpy())
        model_probs["Default Transformer"] = np.concatenate(logits_list)

    # Tuned transformer
    tuned_path = "outputs/transformer_best_tuned.pt"
    if os.path.exists(tuned_path):
        model_tuned = TemporalTransformer(d_model=d_model, n_heads=n_heads,
                                          n_layers=n_layers, d_ff=d_ff,
                                          month_dim=month_dim)
        state = torch.load(tuned_path, map_location=device, weights_only=True)
        model_tuned.load_state_dict(state)
        model_tuned.eval()
        logits_list = []
        with torch.no_grad():
            for demo, monthly, _ in test_loader:
                logits, _ = model_tuned(demo, monthly)
                logits_list.append(torch.sigmoid(logits).numpy())
        model_probs["Tuned Transformer"] = np.concatenate(logits_list)

    # Random Forest
    rf_path = "outputs/rf_best.joblib"
    if os.path.exists(rf_path):
        import joblib
        rf = joblib.load(rf_path)
        model_probs["Random Forest"] = rf.predict_proba(X_te.values)[:, 1]

    if not model_probs:
        print("No model files found. Run models first.")
        return

    # Plot
    colors = {
        "Default Transformer": "#4c9be8",
        "Tuned Transformer":   "#e8754c",
        "Random Forest":       "#50c8a0",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(labels_np, probs)
        roc_auc     = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[name], lw=2,
                 label=f"{name} (AUC={roc_auc:.4f})")

        prec, rec, _ = precision_recall_curve(labels_np, probs)
        pr_auc       = auc(rec, prec)
        ax2.plot(rec, prec, color=colors[name], lw=2,
                 label=f"{name} (AUC={pr_auc:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve — All Models")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    baseline = labels_np.mean()
    ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, lw=1,
                label=f"Baseline (prevalence={baseline:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve — All Models")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/figures/roc_pr_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── 5. Model comparison — THREE models ────────────────────────────────────────

def plot_comparison():
    """
    Compares Default Transformer, Tuned Transformer, and Random Forest
    on Test AUROC and Test Accuracy (tuned threshold).
    """
    results = {}

    for name, fpath in [
        ("Default Transformer", "outputs/transformer_results_default.json"),
        ("Tuned Transformer",   "outputs/transformer_results_tuned.json"),
        ("Random Forest",       "outputs/rf_results.json"),
    ]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {fpath} not found — skipping {name}")

    if len(results) < 2:
        print("Need at least 2 model results to compare.")
        return

    # Use tuned-threshold accuracy for transformers
    for name in results:
        if "test_acc_tuned" in results[name]:
            results[name]["test_acc"] = results[name]["test_acc_tuned"]

    metrics = ["test_auroc", "test_acc"]
    labels  = ["Test AUROC", "Test Accuracy (tuned threshold)"]
    x       = np.arange(len(metrics))
    w       = 0.25
    colors  = ["#4c9be8", "#e8754c", "#50c8a0"]

    fig, ax = plt.subplots(figsize=(9, 5))

    n_models = len(results)
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w

    for i, (name, res) in enumerate(results.items()):
        vals = [res.get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, w, label=name,
                      color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set (Three Models)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = "outputs/figures/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # Print comparison table
    print("\n── Model Comparison ──────────────────────────────────────────")
    print(f"{'Metric':<35} {'Default TF':>12} {'Tuned TF':>10} {'Random Forest':>15}")
    print("-" * 75)
    for m, label in zip(metrics, labels):
        vals = [results.get(name, {}).get(m, float("nan"))
                for name in ["Default Transformer", "Tuned Transformer", "Random Forest"]]
        print(f"{label:<35} {vals[0]:>12.4f} {vals[1]:>10.4f} {vals[2]:>15.4f}")

    # Also print n_params for transformers
    print("\n── Model Size ────────────────────────────────────────────────")
    for name in ["Default Transformer", "Tuned Transformer"]:
        if name in results and "n_params" in results[name]:
            print(f"  {name}: {results[name]['n_params']:,} parameters")
    if "Random Forest" in results and "best_params" in results["Random Forest"]:
        n_trees = results["Random Forest"]["best_params"].get("n_estimators", "?")
        print(f"  Random Forest: {n_trees} trees")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all",
                   choices=["all", "eda", "attention", "curves", "roc_pr", "compare"])
    p.add_argument("--d_model",   type=int, default=128)
    p.add_argument("--n_heads",   type=int, default=8)
    p.add_argument("--n_layers",  type=int, default=3)
    p.add_argument("--d_ff",      type=int, default=128)
    p.add_argument("--month_dim", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    kw = dict(d_model=args.d_model, n_heads=args.n_heads,
              n_layers=args.n_layers, d_ff=args.d_ff, month_dim=args.month_dim)

    if mode in ("all", "eda"):
        print("=== EDA plots ===")
        plot_eda()

    if mode in ("all", "curves"):
        print("\n=== Training curves ===")
        plot_curves()

    if mode in ("all", "attention"):
        if os.path.exists("outputs/transformer_best_tuned.pt"):
            print("\n=== Attention heatmaps ===")
            plot_attention(**kw)
        else:
            print("No tuned transformer checkpoint found.")

    if mode in ("all", "roc_pr"):
        print("\n=== ROC and PR curves ===")
        plot_roc_pr(**kw)

    if mode in ("all", "compare"):
        print("\n=== Model comparison ===")
        plot_comparison()