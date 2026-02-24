"""
Isolation Forest wrapper for one-class anomaly detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from ensemble_ddos_detection.config import IsolationForestConfig


class IsolationForestModel:
    """Wraps sklearn IsolationForest with normalized anomaly scoring."""

    def __init__(self, config: IsolationForestConfig | None = None):
        self.config = config or IsolationForestConfig()
        self.model = SklearnIF(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(self, X_train: np.ndarray) -> "IsolationForestModel":
        """Train on benign-only data."""
        print(f"[IsolationForest] Training on {X_train.shape[0]:,} samples...")
        self.model.fit(X_train)

        # Calibrate score normalization bounds on training data
        raw_scores = self.model.decision_function(X_train)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        print("[IsolationForest] Training complete.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1].

        Higher score = more anomalous.
        sklearn's decision_function is inverted: lower = more anomalous,
        so we flip and normalize.
        """
        raw = self.model.decision_function(X)
        # Invert so that higher = more anomalous
        inverted = -raw
        # Normalize to [0, 1]
        inv_min = -self._score_max
        inv_max = -self._score_min
        if inv_max - inv_min < 1e-10:
            return np.zeros(len(X))
        normalized = (inverted - inv_min) / (inv_max - inv_min)
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 = attack, 0 = benign."""
        return (self.score(X) >= threshold).astype(int)

    @property
    def sklearn_model(self):
        """Access the underlying sklearn model (for ONNX export)."""
        return self.model
