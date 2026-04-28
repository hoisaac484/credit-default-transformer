"""
Data loading and preprocessing for the UCI Credit Card Default dataset.
Downloads via ucimlrepo and returns 70/15/15 train/val/test splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ── Feature groups ─────────────────────────────────────────────────────────────
STATIC_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Monthly columns ordered oldest→most recent (Apr→Sep)
PAY_COLS  = ["PAY_6",     "PAY_5",     "PAY_4",     "PAY_3",     "PAY_2",     "PAY_0"]
BILL_COLS = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
AMT_COLS  = ["PAY_AMT6",  "PAY_AMT5",  "PAY_AMT4",  "PAY_AMT3",  "PAY_AMT2",  "PAY_AMT1"]

TARGET_COL = "default"

# PAY_COLS excluded from scaler — they are ordinal ints used as embedding indices
NUMERIC_STATIC_COLS  = ["LIMIT_BAL", "AGE"]
MONTHLY_NUMERIC_COLS = BILL_COLS + AMT_COLS


def load_and_clean() -> pd.DataFrame:
    """Fetch from UCI repo and apply light cleaning."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = dataset.variables["description"].tolist()[1:]
    except Exception as e:
        raise RuntimeError(
            "Could not fetch dataset. Install ucimlrepo: pip install ucimlrepo\n"
            f"Original error: {e}"
        )

    df = df.rename(columns={"default payment next month": "default"})

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remap undocumented EDUCATION categories (0, 5, 6 → 4 "Others") and
    # MARRIAGE category 0 → 3 "Others" so indices are contiguous and meaningful
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})

    df = df.reset_index(drop=True)
    return df


def split_and_scale(df: pd.DataFrame, seed: int = 42):
    """70/15/15 stratified split + StandardScaler on numeric columns only."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[TARGET_COL]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df[TARGET_COL]
    )

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    scaler_static  = StandardScaler()
    scaler_monthly = StandardScaler()

    train_df[NUMERIC_STATIC_COLS]  = scaler_static.fit_transform(train_df[NUMERIC_STATIC_COLS])
    val_df[NUMERIC_STATIC_COLS]    = scaler_static.transform(val_df[NUMERIC_STATIC_COLS])
    test_df[NUMERIC_STATIC_COLS]   = scaler_static.transform(test_df[NUMERIC_STATIC_COLS])

    train_df[MONTHLY_NUMERIC_COLS] = scaler_monthly.fit_transform(train_df[MONTHLY_NUMERIC_COLS])
    val_df[MONTHLY_NUMERIC_COLS]   = scaler_monthly.transform(val_df[MONTHLY_NUMERIC_COLS])
    test_df[MONTHLY_NUMERIC_COLS]  = scaler_monthly.transform(test_df[MONTHLY_NUMERIC_COLS])

    return train_df, val_df, test_df, scaler_static, scaler_monthly


class CreditDataset(Dataset):
    """
    Returns (X_dict, target) per sample.

    X_dict keys:
      static_num  : (2,)   float32  — scaled LIMIT_BAL, AGE
      static_cat  : (3,)   long     — SEX, EDUCATION, MARRIAGE (embedding indices)
      monthly_num : (6, 2) float32  — scaled BILL_AMT, PAY_AMT per month
      monthly_pay : (6,)   long     — PAY status clamped to [0, 15] (embedding index)
    """

    def __init__(self, X, y=None):
        if hasattr(X, "reset_index"):
            self.X = X.reset_index(drop=True)
        else:
            self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx] if hasattr(self.X, "iloc") else self.X[idx]

        static_num = torch.tensor(
            [row["LIMIT_BAL"], row["AGE"]],
            dtype=torch.float32,
        )

        static_cat = torch.tensor(
            [int(row["SEX"]), int(row["EDUCATION"]), int(row["MARRIAGE"])],
            dtype=torch.long,
        )

        monthly_num = []
        monthly_pay = []
        for p, b, a in zip(PAY_COLS, BILL_COLS, AMT_COLS):
            monthly_num.append([row[b], row[a]])
            monthly_pay.append(max(0, min(15, int(row[p]) + 2)))  # clamp to valid embedding range

        monthly_num = torch.tensor(monthly_num, dtype=torch.float32)   # (6, 2)
        monthly_pay = torch.tensor(monthly_pay, dtype=torch.long)      # (6,)

        X_dict = {
            "static_num":  static_num,
            "static_cat":  static_cat,
            "monthly_num": monthly_num,
            "monthly_pay": monthly_pay,
        }

        if self.y is not None:
            val = self.y.iloc[idx] if hasattr(self.y, "iloc") else self.y[idx]
            target = torch.tensor(float(val), dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        return X_dict, target


def get_dataloaders(batch_size: int = 128, seed: int = 42, num_workers: int = 0):
    """End-to-end: fetch → clean → split → scale → DataLoader."""
    df = load_and_clean()
    train_df, val_df, test_df, scaler_static, scaler_monthly = split_and_scale(df, seed)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val   = val_df.drop(columns=[TARGET_COL])
    y_val   = val_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL]

    train_dataset = CreditDataset(X_train, y_train)
    val_dataset   = CreditDataset(X_val,   y_val)
    test_dataset  = CreditDataset(X_test,  y_test)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_dataset, shuffle=True,  **kw)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **kw)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **kw)

    return (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler_static, scaler_monthly,
    )


if __name__ == "__main__":
    result = get_dataloaders()
    train_loader, val_loader, test_loader = result[0], result[1], result[2]
    train_dataset, val_dataset, test_dataset = result[3], result[4], result[5]

    batch = next(iter(train_loader))
    inputs_dict, targets = batch
    for k, v in inputs_dict.items():
        print(f"{k:12} : {v.shape}")
    print(f"{'target':12} : {targets.shape}")

    print(f"\nTrain: {len(train_dataset):,}  Val: {len(val_dataset):,}  Test: {len(test_dataset):,}")
