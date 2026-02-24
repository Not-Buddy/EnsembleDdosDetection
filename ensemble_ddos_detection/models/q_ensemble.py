"""
Q-Ensemble: Weighted score-level combination of multiple anomaly detectors.

Optimizes per-model weights and the decision threshold on a validation set
to maximize a chosen metric (default: F1-score).
"""

import numpy as np
from itertools import product
from sklearn.metrics import f1_score, precision_score, recall_score
from dataclasses import dataclass

from ensemble_ddos_detection.config import QEnsembleConfig


@dataclass
class EnsembleResult:
    """Holds the optimized ensemble parameters."""
    weights: list[float]       # [w_if, w_ae, w_svm]
    threshold: float
    best_metric_value: float
    metric_name: str


class QEnsemble:
    """
    Q-Ensemble combiner for anomaly detection.

    Combines scores from N anomaly detectors via weighted averaging:
        score_ensemble = Σ(w_i * s_i)  where Σ(w_i) = 1

    Weights and threshold are optimized on a validation set.
    """

    def __init__(self, n_models: int = 3, config: QEnsembleConfig | None = None):
        self.config = config or QEnsembleConfig()
        self.n_models = n_models
        self.weights: np.ndarray = np.ones(n_models) / n_models  # uniform default
        self.threshold: float = 0.5
        self._optimized: bool = False

    def optimize(
        self,
        scores: list[np.ndarray],
        y_true: np.ndarray,
    ) -> EnsembleResult:
        """
        Grid-search over weight simplex and threshold to maximize the target metric.

        Args:
            scores: List of anomaly score arrays, one per model. Each shape (n_samples,).
            y_true: Ground truth binary labels (0=benign, 1=attack).

        Returns:
            EnsembleResult with optimized weights and threshold.
        """
        assert len(scores) == self.n_models, (
            f"Expected {self.n_models} score arrays, got {len(scores)}"
        )

        scores_matrix = np.stack(scores, axis=1)  # (n_samples, n_models)
        steps = self.config.weight_grid_steps

        # ── Generate weight candidates on the simplex ──────────────────
        # For 3 models: w1 + w2 + w3 = 1, w_i >= 0
        weight_candidates: list[tuple[float, ...]] = []
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                w = (i / steps, j / steps, k / steps)
                weight_candidates.append(w)

        # ── Threshold candidates ───────────────────────────────────────
        threshold_candidates = np.linspace(0.05, 0.95, 50)

        best_score = -1.0
        best_weights = self.weights.copy()
        best_threshold = self.threshold

        print(
            f"[Q-Ensemble] Optimizing weights ({len(weight_candidates)} candidates) "
            f"× thresholds ({len(threshold_candidates)} candidates)..."
        )

        for w in weight_candidates:
            w_arr = np.array(w)
            ensemble_scores = scores_matrix @ w_arr  # (n_samples,)

            for thr in threshold_candidates:
                y_pred = (ensemble_scores >= thr).astype(int)

                if self.config.optimize_metric == "f1":
                    metric_val = f1_score(y_true, y_pred, zero_division=0)
                elif self.config.optimize_metric == "precision":
                    metric_val = precision_score(y_true, y_pred, zero_division=0)
                elif self.config.optimize_metric == "recall":
                    metric_val = recall_score(y_true, y_pred, zero_division=0)
                else:
                    metric_val = f1_score(y_true, y_pred, zero_division=0)

                if metric_val > best_score:
                    best_score = metric_val
                    best_weights = w_arr.copy()
                    best_threshold = float(thr)

        self.weights = best_weights
        self.threshold = best_threshold
        self._optimized = True

        print(f"[Q-Ensemble] Optimized weights: IF={self.weights[0]:.3f}, "
              f"AE={self.weights[1]:.3f}, SVM={self.weights[2]:.3f}")
        print(f"[Q-Ensemble] Optimized threshold: {self.threshold:.4f}")
        print(f"[Q-Ensemble] Best {self.config.optimize_metric}: {best_score:.4f}")

        return EnsembleResult(
            weights=self.weights.tolist(),
            threshold=self.threshold,
            best_metric_value=best_score,
            metric_name=self.config.optimize_metric,
        )

    def combine_scores(self, scores: list[np.ndarray]) -> np.ndarray:
        """Compute weighted ensemble score."""
        scores_matrix = np.stack(scores, axis=1)
        return scores_matrix @ self.weights

    def predict(self, scores: list[np.ndarray]) -> np.ndarray:
        """Binary prediction using optimized weights and threshold."""
        ensemble_scores = self.combine_scores(scores)
        return (ensemble_scores >= self.threshold).astype(int)

    def to_dict(self) -> dict:
        """Serialize ensemble config for export."""
        return {
            "n_models": self.n_models,
            "weights": self.weights.tolist(),
            "threshold": self.threshold,
            "model_names": ["isolation_forest", "autoencoder", "one_class_svm"],
            "optimized": self._optimized,
        }
