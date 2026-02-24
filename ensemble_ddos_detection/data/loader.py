"""
Dataset loader for the CIC-DDoS2019 dataset.

Reads all parquet files, concatenates them, and binarizes labels
into 0 (Benign) and 1 (Attack).
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ensemble_ddos_detection.config import (
    DATASET_DIR,
    LABEL_COLUMN,
    BENIGN_LABEL,
    DROP_COLUMNS,
)


def load_dataset(dataset_dir: Path = DATASET_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and merge all CIC-DDoS2019 parquet files.

    Returns:
        X: DataFrame of numeric features (constant columns dropped).
        y: Series of binary labels (0 = Benign, 1 = Attack).
    """
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    print(f"[Loader] Found {len(parquet_files)} parquet files in {dataset_dir}")

    frames: list[pd.DataFrame] = []
    for fp in tqdm(parquet_files, desc="Loading parquets"):
        df = pd.read_parquet(fp)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"[Loader] Total samples: {len(data):,}")

    # ── Extract labels & binarize ──────────────────────────────────────
    if LABEL_COLUMN not in data.columns:
        raise KeyError(f"Label column '{LABEL_COLUMN}' not found in dataset")

    # Convert to string for safe comparison (category dtype)
    labels_raw = data[LABEL_COLUMN].astype(str).str.strip()
    y = (~labels_raw.str.lower().eq(BENIGN_LABEL.lower())).astype(int)

    benign_count = (y == 0).sum()
    attack_count = (y == 1).sum()
    print(f"[Loader] Benign: {benign_count:,} | Attack: {attack_count:,}")

    # ── Drop label + unwanted columns ──────────────────────────────────
    X = data.drop(columns=[LABEL_COLUMN], errors="ignore")

    cols_to_drop = [c for c in DROP_COLUMNS if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        print(f"[Loader] Dropped {len(cols_to_drop)} constant/unwanted columns")

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])
    print(f"[Loader] Final feature count: {X.shape[1]}")

    return X, y
