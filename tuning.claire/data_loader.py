"""
Data loading and preprocessing for the UCI Credit Card Default dataset.
Downloads via ucimlrepo and returns train/val/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ── Feature groups ─────────────────────────────────────────────────────────────
DEMO_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Monthly columns ordered Sep→Apr (most-recent first, matching PAY_0..PAY_6)
PAY_COLS   = ["PAY_0",  "PAY_2",  "PAY_3",  "PAY_4",  "PAY_5",  "PAY_6"]
BILL_COLS  = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
AMT_COLS   = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]
N_MONTHS    = 6   # Sep, Aug, Jul, Jun, May, Apr 2005
MONTH_DIM   = 5   # PAY_t, BILL_AMT_t, PAY_AMT_t, util_ratio_t, pay_ratio_t
TARGET_COL  = "default payment next month"


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch from UCI repo (cached after first call by ucimlrepo)."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
        X = ds.data.features.copy()
        y = ds.data.targets.iloc[:, 0].copy()
    except Exception as e:
        raise RuntimeError(
            "Could not fetch dataset. Install ucimlrepo: pip install ucimlrepo\n"
            f"Original error: {e}"
        )

    # ucimlrepo returns generic names X1-X23; rename to descriptive names
    col_map = {
        "X1":  "LIMIT_BAL",
        "X2":  "SEX",
        "X3":  "EDUCATION",
        "X4":  "MARRIAGE",
        "X5":  "AGE",
        "X6":  "PAY_0",
        "X7":  "PAY_2",
        "X8":  "PAY_3",
        "X9":  "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
    }
    X = X.rename(columns=col_map)
    y = y.rename("default")
    return X, y


def clean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning:
    - EDUCATION has undocumented values 0,5,6 → recode to 4 (other)
    - MARRIAGE has undocumented value 0 → recode to 3 (other)
    """
    X = X.copy()
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X["MARRIAGE"]  = X["MARRIAGE"].replace({0: 3})
    return X


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """80/10/10 stratified split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


class CreditCardDataset(Dataset):
    """
    Returns two tensors per sample that the model concatenates into a length-8 sequence:

      [CLS]      : learnable token added by the model itself (not stored here)
      Token 0    : demographics [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE]  → dim 5
      Tokens 1–6 : monthly     [PAY_t, BILL_AMT_t, PAY_AMT_t,
                                 util_ratio_t, pay_ratio_t]                  → dim 5

    util_ratio_t = BILL_AMT_t / LIMIT_BAL  (credit utilisation, clipped to [-1, 3])
    pay_ratio_t  = PAY_AMT_t  / BILL_AMT_t when bill > 0, else 0 (clipped to [0, 5])

    All values are standardised (scalers fitted on training set only).
    Scalers are stored so they can be applied to val/test without leakage.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 demo_scaler: StandardScaler | None = None,
                 monthly_scaler: StandardScaler | None = None,
                 fit_scalers: bool = False):

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # ── Demographics ──────────────────────────────────────────────────────
        demo_raw = X[DEMO_COLS].values.astype(np.float32)
        if fit_scalers:
            self.demo_scaler = StandardScaler()
            demo_scaled = self.demo_scaler.fit_transform(demo_raw)
        else:
            self.demo_scaler = demo_scaler
            demo_scaled = demo_scaler.transform(demo_raw)

        # ── Monthly tokens ────────────────────────────────────────────────────
        limit = X["LIMIT_BAL"].values.astype(np.float32)   # (N,)

        # Engineered features per month
        bill  = X[BILL_COLS].values.astype(np.float32)     # (N, 6)
        amt   = X[AMT_COLS].values.astype(np.float32)      # (N, 6)

        # Utilisation: bill / limit — clip to [-1, 3] to handle credit balances & overlimit
        util  = bill / (limit[:, np.newaxis] + 1)
        util  = np.clip(util, -1.0, 3.0)

        # Payment ratio: fraction of (positive) bill paid; 0 when no bill outstanding
        bill_pos = np.where(bill > 0, bill, np.nan)
        payr     = np.where(bill > 0, amt / bill_pos, 0.0)
        payr     = np.clip(payr, 0.0, 5.0)                  # cap at 5× (full payoff + credit)

        # Stack into (N, 6, 5): axis-2 = [PAY, BILL, AMT, util, pay_ratio]
        monthly_raw = np.stack([
            X[PAY_COLS].values.astype(np.float32),
            bill,
            amt,
            util,
            payr,
        ], axis=2)                                          # (N, 6, 5)

        N = monthly_raw.shape[0]
        monthly_flat = monthly_raw.reshape(N, -1)           # (N, 30)

        if fit_scalers:
            self.monthly_scaler = StandardScaler()
            monthly_flat_scaled = self.monthly_scaler.fit_transform(monthly_flat)
        else:
            self.monthly_scaler = monthly_scaler
            monthly_flat_scaled = monthly_scaler.transform(monthly_flat)

        monthly_scaled = monthly_flat_scaled.reshape(N, N_MONTHS, MONTH_DIM)  # (N,6,5)

        # ── Assemble token sequence ───────────────────────────────────────────
        # Token 0: demographics → (N, 1, 5)
        demo_tok = demo_scaled[:, np.newaxis, :]           # (N, 1, 5)
        # Tokens 1-6: monthly snapshots → (N, 6, 5)
        # Stored separately; the model projects each type independently then concatenates.
        self.demo_tokens    = torch.from_numpy(demo_tok)       # (N, 1, 5)
        self.monthly_tokens = torch.from_numpy(monthly_scaled) # (N, 6, 5)
        self.labels         = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.demo_tokens[idx],     # (1, 5)
            self.monthly_tokens[idx],  # (6, 5)
            self.labels[idx],          # scalar
        )


def get_dataloaders(batch_size: int = 256, seed: int = 42, num_workers: int = 0):
    """End-to-end: fetch → clean → split → scale → DataLoader."""
    X, y = load_raw()
    X    = clean(X)

    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y, seed)

    train_ds = CreditCardDataset(X_tr, y_tr, fit_scalers=True)
    val_ds   = CreditCardDataset(X_val, y_val,
                                  demo_scaler=train_ds.demo_scaler,
                                  monthly_scaler=train_ds.monthly_scaler)
    test_ds  = CreditCardDataset(X_te, y_te,
                                  demo_scaler=train_ds.demo_scaler,
                                  monthly_scaler=train_ds.monthly_scaler)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # Positive class weight for imbalanced BCE loss
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    return train_loader, val_loader, test_loader, pos_weight, (X_tr, X_val, X_te, y_tr, y_val, y_te)


if __name__ == "__main__":
    tr, val, te, pw, _ = get_dataloaders()
    demo, monthly, labels = next(iter(tr))
    print("demo shape   :", demo.shape)      # (B, 1, 5)
    print("monthly shape:", monthly.shape)   # (B, 6, 5)
    print("labels shape :", labels.shape)    # (B,)
    print("pos_weight   :", pw.item())
