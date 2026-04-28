"""
Random Forest benchmarks for credit-card default prediction.

Two variants:
  1. RF on raw (scaled) features — GridSearchCV, 5-fold, f1_macro
  2. RF on tokenised embeddings  — same grid, using CreditTokenEmbedding

Both use class_weight="balanced" to match the pos_weight treatment given to the SLM,
and both are evaluated at threshold 0.5 AND 0.3 to allow a fair comparison.

Usage:
    python random_forest.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)

from data_loader import get_dataloaders
from transformer_model import CreditTokenEmbedding


def get_tokens(dataloader: torch.utils.data.DataLoader, tokeniser: CreditTokenEmbedding) -> np.ndarray:
    """Run dataloader through tokeniser and return flattened embeddings."""
    all_tokens = []
    tokeniser.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            tokens = tokeniser(
                inputs["static_num"],
                inputs["static_cat"],
                inputs["monthly_num"],
                inputs["monthly_pay"],
            )
            flattened = tokens.view(tokens.size(0), -1)
            all_tokens.append(flattened.numpy())
    return np.concatenate(all_tokens, axis=0)


def main():
    os.makedirs("output", exist_ok=True)

    (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler_static, scaler_monthly,
    ) = get_dataloaders(batch_size=128, seed=42)

    param_grid = {
        "n_estimators":      [75, 100, 125],
        "max_depth":         [15, 20, 25],
        "min_samples_split": [8, 10, 12],
    }

    # ── RF on raw data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Random Forest — Raw Data")
    print("=" * 60)

    # class_weight="balanced" mirrors the pos_weight treatment given to the SLM
    grid_raw = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_raw.fit(X_train, y_train.values.ravel())

    print(f"Best Parameters: {grid_raw.best_params_}")
    print(f"Best CV Score (F1): {grid_raw.best_score_:.4f}")

    best_rf   = grid_raw.best_estimator_
    probs_raw = best_rf.predict_proba(X_test)[:, 1]
    preds_raw = best_rf.predict(X_test)                          # threshold 0.5
    preds_raw_cons = (probs_raw >= 0.3).astype(int)              # threshold 0.3

    print("\n--- Classification Report (threshold 0.5) ---")
    print(classification_report(y_test, preds_raw, target_names=["No Default", "Default"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs_raw):.4f}")

    print("\n--- Classification Report (threshold 0.3) ---")
    print(classification_report(y_test, preds_raw_cons, target_names=["No Default", "Default"]))

    # Confusion matrix — threshold 0.5
    cm_raw = confusion_matrix(y_test, preds_raw)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", linewidths=0,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix — Random Forest (Raw Data, Threshold 0.5)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("output/RF_raw_confusion_matrix.png")
    plt.close()

    # Confusion matrix — threshold 0.3
    cm_raw_cons = confusion_matrix(y_test, preds_raw_cons)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_raw_cons, annot=True, fmt="d", cmap="Reds", linewidths=0,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix — Random Forest (Raw Data, Threshold 0.3)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("output/RF_raw_confusion_matrix_conservative.png")
    plt.close()

    feat_series = pd.Series(best_rf.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10, 6))
    feat_series.nlargest(10).plot(kind="barh", color="skyblue")
    plt.title("Top 10 Most Important Features (Random Forest)")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    plt.savefig("output/RF_raw_feature_importance.png")
    plt.close()

    joblib.dump(best_rf, "output/raw_data_RF_model.joblib")

    # ── RF on tokenised data ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Random Forest — Tokenised Data")
    print("=" * 60)

    torch.manual_seed(42)
    d_model   = 64
    tokeniser = CreditTokenEmbedding(d_model)

    X_train_tok = get_tokens(train_loader, tokeniser)
    X_test_tok  = get_tokens(test_loader,  tokeniser)

    grid_tok = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_tok.fit(X_train_tok, y_train.values.ravel())

    print(f"Best Parameters: {grid_tok.best_params_}")
    print(f"Best CV Score (F1): {grid_tok.best_score_:.4f}")

    best_rf_tok  = grid_tok.best_estimator_
    probs_tok    = best_rf_tok.predict_proba(X_test_tok)[:, 1]
    preds_tok    = best_rf_tok.predict(X_test_tok)              # threshold 0.5
    preds_tok_cons = (probs_tok >= 0.3).astype(int)             # threshold 0.3

    print("\n--- Classification Report (threshold 0.5) ---")
    print(classification_report(y_test, preds_tok, target_names=["No Default", "Default"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs_tok):.4f}")

    cm_tok = confusion_matrix(y_test, preds_tok)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_tok, annot=True, fmt="d", cmap="Oranges", linewidths=0,
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title("Confusion Matrix — Random Forest (Tokenised Data, Threshold 0.5)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("output/RF_tokenised_confusion_matrix.png")
    plt.close()

    token_labels = (
        ["LIMIT_BAL", "SEX", "EDU", "MARRIAGE", "AGE"]
        + ["Month 1 (Oldest)", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6 (Recent)"]
    )
    raw_imp = best_rf_tok.feature_importances_
    agg_imp = [raw_imp[i * d_model:(i + 1) * d_model].sum() for i in range(len(token_labels))]

    tok_series = pd.Series(agg_imp, index=token_labels).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    tok_series.plot(kind="barh", color="teal", edgecolor="black")
    plt.title("Importance by Logical Token (Aggregated)")
    plt.xlabel("Total Gini Importance (Summed across Dimensions)")
    plt.ylabel("Token Type")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("output/RF_tokenised_feature_importance.png")
    plt.close()

    joblib.dump(best_rf_tok, "output/tokenised_data_RF_model.joblib")

    print("\nArtefacts saved to output/")


if __name__ == "__main__":
    main()
