"""
Isolation Forest wrapper for one-class anomaly detection.

Uses percentile-based normalization for better score separation.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from ensemble_ddos_detection.config import IsolationForestConfig


class IsolationForestModel:
    """Wraps sklearn IsolationForest with percentile-normalized anomaly scoring."""

    def __init__(self, config: IsolationForestConfig | None = None):
        self.config = config or IsolationForestConfig()
        self.model = SklearnIF(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self._p1: float = 0.0    # 1st percentile of training scores
        self._p99: float = 1.0   # 99th percentile of training scores

    def fit(self, X_train: np.ndarray) -> "IsolationForestModel":
        """Train on benign-only data."""
        print(f"[IsolationForest] Training on {X_train.shape[0]:,} samples...")
        self.model.fit(X_train)

        # Calibrate percentile bounds on training data using score_samples
        raw_scores = self.model.score_samples(X_train)
        # score_samples: lower = more anomalous; we invert later
        self._p1 = float(np.percentile(raw_scores, 1))
        self._p99 = float(np.percentile(raw_scores, 99))
        print(f"[IsolationForest] Score bounds: P1={self._p1:.4f}, P99={self._p99:.4f}")
        print("[IsolationForest] Training complete.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1].

        Higher score = more anomalous.
        Uses score_samples() with percentile-based normalization for
        better separation between benign and attack traffic.
        """
        raw = self.model.score_samples(X)

        # Invert: lower score_samples = more anomalous → higher output
        inverted = -raw
        inv_p1 = -self._p99    # maps to low anomaly score
        inv_p99 = -self._p1    # maps to high anomaly score

        denom = inv_p99 - inv_p1
        if denom < 1e-10:
            return np.zeros(len(X))

        normalized = (inverted - inv_p1) / denom
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 = attack, 0 = benign."""
        return (self.score(X) >= threshold).astype(int)

    @property
    def sklearn_model(self):
        """Access the underlying sklearn model (for ONNX export)."""
        return self.model
