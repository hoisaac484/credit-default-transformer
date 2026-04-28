"""
Training loop for the TemporalTransformer using skorch + RandomizedSearchCV.

Usage:
    python train_transformer.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, GradientNormClipping
from skorch.helper import predefined_split

from data_loader import get_dataloaders, CreditDataset, TARGET_COL
from transformer_model import TemporalTransformer


def main():
    os.makedirs("output", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler_static, scaler_monthly,
    ) = get_dataloaders(batch_size=128, seed=42)

    y_series = y_train.values.astype("float32")

    # Positive class weight from actual class ratio
    pos_weight = torch.tensor(
        [(y_series == 0).sum() / (y_series == 1).sum()]
    ).to(device)

    # ── skorch NeuralNetClassifier ────────────────────────────────────────────
    net = NeuralNetClassifier(
        TemporalTransformer,
        dataset=CreditDataset,
        criterion=nn.BCEWithLogitsLoss,
        criterion__pos_weight=pos_weight,
        optimizer=optim.Adam,
        lr=0.001,
        max_epochs=50,
        batch_size=128,
        device=device,
        train_split=predefined_split(val_dataset),
        callbacks=[
            EarlyStopping(patience=5, monitor="valid_loss"),
            GradientNormClipping(gradient_clip_value=1.0),
        ],
        verbose=1,
    )

    # ── Hyperparameter search ─────────────────────────────────────────────────
    shared_params = {
        "module__n_layers": [2, 3],
        "module__d_ff":     [64, 128],
        "module__dropout":  [0.1, 0.2],
        "lr":               [0.001, 0.0001],
    }
    param_grid = [
        {"module__d_model": [32], "module__n_heads": [2], **shared_params},
        {"module__d_model": [64], "module__n_heads": [4], **shared_params},
    ]

    grid = RandomizedSearchCV(
        net, param_grid,
        n_iter=20, cv=3,
        scoring="f1_macro",
        random_state=42,
        refit=True,
    )
    grid.fit(X_train, y_series)

    print(f"\nBest Score : {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")

    # ── Evaluation on test set ─────────────────────────────────────────────────
    y_probs = grid.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred  = grid.best_estimator_.predict(X_test)

    print("\n--- Classification Report (threshold 0.5) ---")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC: {auc:.4f}")

    # Conservative threshold
    custom_threshold = 0.3
    y_pred_conservative = (y_probs > custom_threshold).astype(int)
    print(f"\n--- Classification Report (threshold {custom_threshold}) ---")
    print(classification_report(
        y_test, y_pred_conservative, target_names=["No Default", "Default"]
    ))

    # ── Save model weights ────────────────────────────────────────────────────
    grid.best_estimator_.save_params(f_params="output/SLM_model_weights.pt")

    # ── Training loss curve ───────────────────────────────────────────────────
    history = grid.best_estimator_.history
    plt.figure(figsize=(10, 5))
    plt.plot(history[:, "train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(history[:, "valid_loss"], label="Val Loss",   color="orange",    lw=2, linestyle="--")
    plt.title("Model Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/SLM_loss.png")
    plt.close()

    # ── Confusion matrix (0.5) ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix — SLM (Threshold 0.5)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("output/SLM_confusion_matrix.png")
    plt.close()

    # ── Confusion matrix (conservative) ──────────────────────────────────────
    cm_cons = confusion_matrix(y_test, y_pred_conservative)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_cons, annot=True, fmt="d", cmap="Reds", linewidths=0,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix — SLM Conservative Threshold ({custom_threshold})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("output/SLM_confusion_matrix_conservative.png")
    plt.close()

    # ── Attention heatmap ─────────────────────────────────────────────────────
    batch = next(iter(test_loader))
    inputs_dict, _ = batch
    s_num = inputs_dict["static_num"].to(device)
    s_cat = inputs_dict["static_cat"].to(device)
    m_num = inputs_dict["monthly_num"].to(device)
    m_pay = inputs_dict["monthly_pay"].to(device)

    attn_maps = grid.best_estimator_.module_.get_attention_maps(s_num, s_cat, m_num, m_pay)
    heatmap_data = attn_maps[0].mean(dim=(0, 1)).detach().cpu().numpy()

    labels = ["LIMIT_BAL", "SEX", "EDU", "MARRIAGE", "AGE"] + [f"Month {i}" for i in range(1, 7)]
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.title("Attention Map: Average over all samples and heads (Layer 1)")
    plt.tight_layout()
    plt.savefig("output/SLM_attention.png", bbox_inches="tight")
    plt.close()

    print("\nArtefacts saved to output/")
    return grid


if __name__ == "__main__":
    main()
