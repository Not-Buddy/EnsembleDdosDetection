"""
Data preprocessor: feature engineering, cleaning, scaling, and splitting.

Feature engineering pipeline:
  1. Log-transform highly skewed features
  2. Drop zero-variance columns
  3. Handle inf/NaN
  4. Mutual information feature selection
  5. StandardScaler (fit on benign-only training data)

One-class training strategy:
  - Train set: benign samples ONLY (models learn "normal")
  - Val / Test sets: mixed (benign + attack) for evaluation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from ensemble_ddos_detection.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SKEW_THRESHOLD,
    MI_DROP_PERCENTILE,
)


@dataclass
class DataSplits:
    """Holds all data splits, fitted scaler, and feature transform metadata."""

    X_train: np.ndarray          # benign-only, scaled
    X_val: np.ndarray            # mixed, scaled
    y_val: np.ndarray
    X_test: np.ndarray           # mixed, scaled
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    n_features: int
    # Feature engineering metadata (for Rust inference)
    log_transformed_columns: list[str] = field(default_factory=list)
    dropped_mi_columns: list[str] = field(default_factory=list)


def _log_transform_skewed(
    X: pd.DataFrame,
    skew_threshold: float = SKEW_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply signed log1p transform to highly skewed features.

    Transform: sign(x) * log1p(|x|)
    This compresses extreme tails while preserving sign.

    Returns:
        Transformed DataFrame and list of transformed column names.
    """
    skewness = X.skew()
    skewed_cols = skewness[skewness.abs() > skew_threshold].index.tolist()

    if skewed_cols:
        X = X.copy()
        for col in skewed_cols:
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
        print(f"[Preprocessor] Log-transformed {len(skewed_cols)} skewed features (|skew| > {skew_threshold})")

    return X, skewed_cols


def _mutual_info_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    drop_percentile: float = MI_DROP_PERCENTILE,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Drop features with low mutual information with the target.

    Returns:
        Filtered X, kept feature names, dropped feature names.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    threshold = np.percentile(mi_scores, drop_percentile)

    keep_mask = mi_scores > threshold
    kept_names = [f for f, k in zip(feature_names, keep_mask) if k]
    dropped_names = [f for f, k in zip(feature_names, keep_mask) if not k]

    if dropped_names:
        print(f"[Preprocessor] MI feature selection: dropped {len(dropped_names)} low-MI features "
              f"(MI < {threshold:.4f})")
        for name in dropped_names:
            idx = feature_names.index(name)
            print(f"    Dropped: {name} (MI={mi_scores[idx]:.4f})")

    return X[:, keep_mask], kept_names, dropped_names


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> DataSplits:
    """
    Full preprocessing pipeline for one-class training.

    Steps:
        1. Log-transform highly skewed features
        2. Drop zero-variance columns
        3. Replace inf → NaN, impute with median
        4. Mutual information feature selection
        5. Split benign vs attack → train/val/test
        6. StandardScaler fitted on train (benign-only)
    """
    print("[Preprocessor] Starting preprocessing...")

    # ── 1. Log-transform skewed features ───────────────────────────────
    X, log_cols = _log_transform_skewed(X)

    # ── 2. Drop zero-variance columns ──────────────────────────────────
    variances = X.var()
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        print(f"[Preprocessor] Dropped {len(zero_var_cols)} zero-variance columns")

    feature_names = X.columns.tolist()
    X_np = X.values.astype(np.float64)
    y_np = y.values

    # ── 3. Handle inf / NaN ────────────────────────────────────────────
    X_np[~np.isfinite(X_np)] = np.nan
    col_medians = np.nanmedian(X_np, axis=0)
    nan_mask = np.isnan(X_np)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X_np[inds] = col_medians[inds[1]]
        print(f"[Preprocessor] Imputed {nan_mask.sum()} NaN/inf values with median")

    # ── 4. Mutual information feature selection ────────────────────────
    X_np, feature_names, dropped_mi_cols = _mutual_info_selection(
        X_np, y_np, feature_names, random_state=random_state
    )

    # ── 5. Separate benign and attack ──────────────────────────────────
    benign_mask = y_np == 0
    X_benign = X_np[benign_mask]
    X_attack = X_np[~benign_mask]
    y_attack = y_np[~benign_mask]

    print(f"[Preprocessor] Benign samples: {len(X_benign):,}")
    print(f"[Preprocessor] Attack samples: {len(X_attack):,}")

    # ── 6. Split benign: train / val / test ────────────────────────────
    val_test_ratio = (VAL_RATIO + TEST_RATIO) / (TRAIN_RATIO + VAL_RATIO + TEST_RATIO)
    test_of_rem = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    X_train, X_benign_rem = train_test_split(
        X_benign, test_size=val_test_ratio, random_state=random_state
    )
    X_val_benign, X_test_benign = train_test_split(
        X_benign_rem, test_size=test_of_rem, random_state=random_state
    )

    # ── 7. Split attack: val / test (50/50) ────────────────────────────
    X_val_attack, X_test_attack, y_val_attack, y_test_attack = train_test_split(
        X_attack, y_attack, test_size=0.5, random_state=random_state
    )

    # ── 8. Combine val and test sets ───────────────────────────────────
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

    # ── 9. Fit scaler on train (benign only) ───────────────────────────
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
        log_transformed_columns=log_cols,
        dropped_mi_columns=dropped_mi_cols,
    )
