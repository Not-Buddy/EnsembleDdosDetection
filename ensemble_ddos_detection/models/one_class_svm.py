"""
One-Class SVM wrapper for one-class anomaly detection.

Uses RBF kernel with subsampling for scalability (SVM has O(n²-n³) cost).
"""

import numpy as np
from sklearn.svm import OneClassSVM as SklearnOCSVM

from ensemble_ddos_detection.config import OneClassSVMConfig


class OneClassSVMModel:
    """Wraps sklearn OneClassSVM with normalized anomaly scoring."""

    def __init__(self, config: OneClassSVMConfig | None = None):
        self.config = config or OneClassSVMConfig()
        self.model = SklearnOCSVM(
            kernel=self.config.kernel,
            gamma=self.config.gamma,
            nu=self.config.nu,
        )
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._subsample_indices: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "OneClassSVMModel":
        """
        Train on benign-only data. Subsamples if needed for scalability.
        """
        n_samples = X_train.shape[0]
        max_samples = self.config.max_samples

        if n_samples > max_samples:
            rng = np.random.RandomState(self.config.random_state)
            self._subsample_indices = rng.choice(n_samples, max_samples, replace=False)
            X_fit = X_train[self._subsample_indices]
            print(
                f"[OneClassSVM] Subsampled {max_samples:,} / {n_samples:,} "
                f"samples for training (SVM scalability)"
            )
        else:
            X_fit = X_train

        print(f"[OneClassSVM] Training on {X_fit.shape[0]:,} samples...")
        self.model.fit(X_fit)

        # Calibrate score normalization bounds
        raw_scores = self.model.decision_function(X_fit)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        print("[OneClassSVM] Training complete.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores normalized to [0, 1].

        Higher score = more anomalous.
        sklearn's decision_function: higher = more normal → we invert.
        """
        raw = self.model.decision_function(X)
        inverted = -raw
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
