"""
Q-Ensemble: Weighted score-level combination of multiple anomaly detectors.

Optimizes per-model weights and the decision threshold on a validation set
to maximize macro-averaged F-beta score (default β=0.5, penalizes false positives).

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


def _vectorized_macro_fbeta(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Compute macro-averaged F-beta scores for multiple prediction sets at once.

    Macro average = mean of per-class F-beta, so both benign and attack
    performance matter equally. β<1 penalizes false positives more
    (improves benign recall / precision of attack class).

    Args:
        y_true: (n_samples,) ground truth
        preds_matrix: (n_samples, n_candidates) binary predictions
        beta: F-beta parameter (<1 favors precision, >1 favors recall)

    Returns:
        (n_candidates,) array of macro F-beta scores
    """
    positives = y_true.astype(bool)
    beta_sq = beta ** 2

    # ── Attack class (positive = 1) ────────────────────────────────
    tp_atk = (preds_matrix & positives[:, None]).sum(axis=0).astype(np.float64)
    fp_atk = (preds_matrix & ~positives[:, None]).sum(axis=0).astype(np.float64)
    fn_atk = (~preds_matrix & positives[:, None]).sum(axis=0).astype(np.float64)

    prec_atk = np.divide(tp_atk, tp_atk + fp_atk, out=np.zeros_like(tp_atk), where=(tp_atk + fp_atk) > 0)
    rec_atk = np.divide(tp_atk, tp_atk + fn_atk, out=np.zeros_like(tp_atk), where=(tp_atk + fn_atk) > 0)
    denom_atk = beta_sq * prec_atk + rec_atk
    fb_atk = np.divide((1 + beta_sq) * prec_atk * rec_atk, denom_atk,
                        out=np.zeros_like(denom_atk), where=denom_atk > 0)

    # ── Benign class (positive = 0, so invert predictions) ─────────
    tp_ben = (~preds_matrix & ~positives[:, None]).sum(axis=0).astype(np.float64)
    fp_ben = (~preds_matrix & positives[:, None]).sum(axis=0).astype(np.float64)
    fn_ben = (preds_matrix & ~positives[:, None]).sum(axis=0).astype(np.float64)

    prec_ben = np.divide(tp_ben, tp_ben + fp_ben, out=np.zeros_like(tp_ben), where=(tp_ben + fp_ben) > 0)
    rec_ben = np.divide(tp_ben, tp_ben + fn_ben, out=np.zeros_like(tp_ben), where=(tp_ben + fn_ben) > 0)
    denom_ben = beta_sq * prec_ben + rec_ben
    fb_ben = np.divide((1 + beta_sq) * prec_ben * rec_ben, denom_ben,
                        out=np.zeros_like(denom_ben), where=denom_ben > 0)

    # ── Macro average ──────────────────────────────────────────────
    return (fb_atk + fb_ben) / 2.0


class QEnsemble:
    """
    Q-Ensemble combiner for anomaly detection.

    Combines scores from N anomaly detectors via weighted averaging:
        score_ensemble = Σ(w_i * s_i)  where Σ(w_i) = 1

    Weights and threshold are optimized on a validation set using
    macro-averaged F-beta to balance both benign and attack performance.
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
        to maximize macro-averaged F-beta score.

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
        beta = self.config.beta
        min_benign_recall = self.config.min_benign_recall

        # Pre-compute benign mask for constraint checking
        benign_mask = ~y_true.astype(bool)  # True where benign
        n_benign = benign_mask.sum()

        # ── Generate weight candidates on the simplex ──────────────────
        weight_candidates: list[tuple[float, ...]] = []
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                weight_candidates.append((i / steps, j / steps, k / steps))
        weights_arr = np.array(weight_candidates)  # (n_weights, n_models)

        # ── Threshold candidates (fine-grained for better benign recall) ──
        thresholds = np.unique(np.concatenate([
            np.linspace(0.01, 0.30, 100),   # fine-grained in critical overlap zone
            np.linspace(0.30, 0.95, 50),     # coarser in the high-confidence zone
        ]))

        n_candidates = len(weight_candidates)
        print(
            f"[Q-Ensemble] Optimizing: {n_candidates} weight combos "
            f"× {len(thresholds)} thresholds (macro F-beta, β={beta}, "
            f"min_benign_recall={min_benign_recall})..."
        )

        best_score = -1.0
        best_weights = self.weights.copy()
        best_threshold = self.threshold

        # Process each weight candidate; vectorize across all thresholds
        for idx, w in enumerate(weights_arr):
            ensemble_scores = scores_matrix @ w  # (n_samples,)

            # Broadcast: (n_samples, 1) >= (1, n_thresholds) → (n_samples, n_thresholds)
            preds = (ensemble_scores[:, None] >= thresholds[None, :])

            # ── Benign recall constraint ───────────────────────────────
            # Benign recall = fraction of benign correctly predicted as 0 (not flagged)
            benign_correct = (~preds[benign_mask, :]).sum(axis=0).astype(np.float64)
            benign_recall = benign_correct / max(n_benign, 1)
            # Mask out thresholds that violate the constraint
            valid_mask = benign_recall >= min_benign_recall

            if not valid_mask.any():
                continue  # no valid threshold for this weight combo

            metric_vals = _vectorized_macro_fbeta(y_true, preds, beta=beta)
            # Zero out invalid thresholds
            metric_vals[~valid_mask] = -1.0

            best_idx = metric_vals.argmax()

            if metric_vals[best_idx] > best_score:
                best_score = metric_vals[best_idx]
                best_weights = w.copy()
                best_threshold = float(thresholds[best_idx])

        self.weights = best_weights
        self.threshold = best_threshold
        self._optimized = True

        print(f"[Q-Ensemble] Optimized weights: IF={self.weights[0]:.3f}, "
              f"AE={self.weights[1]:.3f}, SVM={self.weights[2]:.3f}")
        print(f"[Q-Ensemble] Optimized threshold: {self.threshold:.4f}")
        print(f"[Q-Ensemble] Best macro F-beta (β={beta}): {best_score:.4f}")

        return EnsembleResult(
            weights=self.weights.tolist(),
            threshold=self.threshold,
            best_metric_value=best_score,
            metric_name=f"macro_fbeta_b{beta}",
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
            "beta": self.config.beta,
            "model_names": ["isolation_forest", "autoencoder", "one_class_svm"],
            "optimized": self._optimized,
        }
