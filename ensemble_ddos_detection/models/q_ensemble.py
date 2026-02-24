"""
Q-Ensemble: Weighted score-level combination of multiple anomaly detectors.

Optimizes per-model weights and the decision threshold on a validation set
to maximize a chosen metric (default: F1-score).

Uses fully vectorized numpy operations for fast grid search.
"""

import numpy as np
from dataclasses import dataclass

from ensemble_ddos_detection.config import QEnsembleConfig


@dataclass
class EnsembleResult:
    """Holds the optimized ensemble parameters."""
    weights: list[float]       # [w_if, w_ae, w_svm]
    threshold: float
    best_metric_value: float
    metric_name: str


def _vectorized_f1(y_true: np.ndarray, preds_matrix: np.ndarray) -> np.ndarray:
    """
    Compute F1 scores for multiple prediction sets at once.

    Args:
        y_true: (n_samples,) ground truth
        preds_matrix: (n_samples, n_candidates) binary predictions

    Returns:
        (n_candidates,) array of F1 scores
    """
    positives = y_true.astype(bool)
    tp = (preds_matrix & positives[:, None]).sum(axis=0).astype(np.float64)
    fp = (preds_matrix & ~positives[:, None]).sum(axis=0).astype(np.float64)
    fn = (~preds_matrix & positives[:, None]).sum(axis=0).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return f1


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
        Vectorized grid-search over weight simplex and thresholds
        to maximize F1-score (or other metric).

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
        weight_candidates: list[tuple[float, ...]] = []
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                weight_candidates.append((i / steps, j / steps, k / steps))
        weights_arr = np.array(weight_candidates)  # (n_weights, n_models)

        # ── Threshold candidates ───────────────────────────────────────
        thresholds = np.linspace(0.05, 0.95, 50)

        n_candidates = len(weight_candidates)
        print(
            f"[Q-Ensemble] Optimizing: {n_candidates} weight combos "
            f"× {len(thresholds)} thresholds (vectorized)..."
        )

        best_score = -1.0
        best_weights = self.weights.copy()
        best_threshold = self.threshold

        # Process each weight candidate; vectorize across all thresholds
        for idx, w in enumerate(weights_arr):
            ensemble_scores = scores_matrix @ w  # (n_samples,)

            # Broadcast: (n_samples, 1) >= (1, n_thresholds) → (n_samples, n_thresholds)
            preds = (ensemble_scores[:, None] >= thresholds[None, :])

            f1s = _vectorized_f1(y_true, preds)
            best_idx = f1s.argmax()

            if f1s[best_idx] > best_score:
                best_score = f1s[best_idx]
                best_weights = w.copy()
                best_threshold = float(thresholds[best_idx])

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
