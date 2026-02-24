"""
Data preprocessor: cleaning, scaling, and train/val/test splitting.

One-class training strategy:
  - Train set: benign samples ONLY (models learn "normal")
  - Val / Test sets: mixed (benign + attack) for evaluation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ensemble_ddos_detection.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


@dataclass
class DataSplits:
    """Holds all data splits and the fitted scaler."""

    X_train: np.ndarray          # benign-only, scaled
    X_val: np.ndarray            # mixed, scaled
    y_val: np.ndarray
    X_test: np.ndarray           # mixed, scaled
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    n_features: int


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> DataSplits:
    """
    Clean, scale, and split the dataset for one-class training.

    Steps:
        1. Drop zero-variance columns
        2. Replace inf → NaN, impute with median
        3. Split benign vs attack
        4. Split benign into train / val_benign / test_benign
        5. Combine val_benign+val_attack → val, test_benign+test_attack → test
        6. StandardScaler fitted on train (benign-only)
    """
    print("[Preprocessor] Starting preprocessing...")

    # ── 1. Drop zero-variance columns ──────────────────────────────────
    variances = X.var()
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        print(f"[Preprocessor] Dropped {len(zero_var_cols)} zero-variance columns")

    feature_names = X.columns.tolist()
    X_np = X.values.astype(np.float64)
    y_np = y.values

    # ── 2. Handle inf / NaN ────────────────────────────────────────────
    X_np[~np.isfinite(X_np)] = np.nan
    col_medians = np.nanmedian(X_np, axis=0)
    nan_mask = np.isnan(X_np)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X_np[inds] = col_medians[inds[1]]
        print(f"[Preprocessor] Imputed {nan_mask.sum()} NaN/inf values with median")

    # ── 3. Separate benign and attack ──────────────────────────────────
    benign_mask = y_np == 0
    X_benign = X_np[benign_mask]
    X_attack = X_np[~benign_mask]
    y_attack = y_np[~benign_mask]

    print(f"[Preprocessor] Benign samples: {len(X_benign):,}")
    print(f"[Preprocessor] Attack samples: {len(X_attack):,}")

    # ── 4. Split benign: train / val / test ────────────────────────────
    val_test_ratio = (VAL_RATIO + TEST_RATIO) / (TRAIN_RATIO + VAL_RATIO + TEST_RATIO)
    test_of_rem = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    X_train, X_benign_rem = train_test_split(
        X_benign, test_size=val_test_ratio, random_state=random_state
    )
    X_val_benign, X_test_benign = train_test_split(
        X_benign_rem, test_size=test_of_rem, random_state=random_state
    )

    # ── 5. Split attack: val / test (50/50) ────────────────────────────
    X_val_attack, X_test_attack, y_val_attack, y_test_attack = train_test_split(
        X_attack, y_attack, test_size=0.5, random_state=random_state
    )

    # ── 6. Combine val and test sets ───────────────────────────────────
    X_val = np.vstack([X_val_benign, X_val_attack])
    y_val = np.concatenate([
        np.zeros(len(X_val_benign), dtype=int),
        y_val_attack,
    ])

    X_test = np.vstack([X_test_benign, X_test_attack])
    y_test = np.concatenate([
        np.zeros(len(X_test_benign), dtype=int),
        y_test_attack,
    ])

    # ── 7. Fit scaler on train (benign only) ───────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[Preprocessor] Train (benign-only): {X_train.shape}")
    print(f"[Preprocessor] Val (mixed):         {X_val.shape} — {y_val.sum()} attacks")
    print(f"[Preprocessor] Test (mixed):        {X_test.shape} — {y_test.sum()} attacks")
    print(f"[Preprocessor] Features:            {len(feature_names)}")

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_names=feature_names,
        n_features=len(feature_names),
    )
