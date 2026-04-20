"""
Random Forest benchmark with hyperparameter tuning via RandomizedSearchCV.

Usage:
    python random_forest.py
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

from data_loader import load_raw, clean, split_data


def prepare_flat(X_tr, X_val, X_te):
    """
    Return feature arrays for the RF.
    Random forests are invariant to monotone feature transformations so no
    scaling is applied — it would have zero effect on splits or predictions.
    """
    return X_tr.values, X_val.values, X_te.values


def tune_and_train(X_tr, y_tr, seed=42, n_iter=30):
    """RandomizedSearchCV over RF hyperparameters."""
    param_dist = {
        "n_estimators"     : [100, 200, 300, 500],
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2", 0.5],
        "class_weight"     : ["balanced", "balanced_subsample", None],
    }

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tr, y_tr)
    print(f"\nBest CV AUROC  : {search.best_score_:.4f}")
    print(f"Best params    : {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_rf(model, X, y, split_name: str):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y, preds)
    auroc = roc_auc_score(y, probs)
    print(f"\n── {split_name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUROC    : {auroc:.4f}")
    print(classification_report(y, preds, target_names=["No Default", "Default"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
    return acc, auroc


def main(seed=42, n_iter=30):
    X, y = load_raw()
    X    = clean(X)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y, seed)

    X_tr_s, X_val_s, X_te_s = prepare_flat(X_tr, X_val, X_te)

    print("=== Random Forest Hyperparameter Search ===")
    best_rf, best_params, best_cv_auroc = tune_and_train(X_tr_s, y_tr, seed, n_iter)

    val_acc,  val_auroc  = evaluate_rf(best_rf, X_val_s, y_val,  "Validation")
    test_acc, test_auroc = evaluate_rf(best_rf, X_te_s,  y_te,   "Test")

    # Feature importances
    importances = pd.Series(
        best_rf.feature_importances_, index=X_tr.columns
    ).sort_values(ascending=False)
    print("\nTop-10 feature importances:")
    print(importances.head(10).to_string())

    # Save results and model
    os.makedirs("outputs", exist_ok=True)
    results = {
        "val_acc"      : val_acc,
        "val_auroc"    : val_auroc,
        "test_acc"     : test_acc,
        "test_auroc"   : test_auroc,
        "best_cv_auroc": best_cv_auroc,
        "best_params"  : {k: str(v) for k, v in best_params.items()},
    }
    with open("outputs/rf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    importances.to_csv("outputs/rf_feature_importances.csv")

    # Persist the fitted model so visualise.py can load it directly
    # rather than refitting from hyperparameters (which would be fragile)
    joblib.dump(best_rf, "outputs/rf_best.joblib")

    print("\nResults saved to outputs/  (model: outputs/rf_best.joblib)")
    return best_rf, results, importances


if __name__ == "__main__":
    main()
