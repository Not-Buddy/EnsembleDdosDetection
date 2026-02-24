"""
Q-Ensemble: Learned stacking combiner for multiple anomaly detectors.

Uses Logistic Regression to learn how to combine scores from N anomaly
detectors, then tunes the decision threshold to maximize macro-averaged
F-beta subject to a minimum benign recall constraint.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

from ensemble_ddos_detection.config import QEnsembleConfig


@dataclass
class EnsembleResult:
    """Holds the optimized ensemble parameters."""
    coefficients: list[float]   # LR coefficients [w_if, w_ae, w_svm]
    intercept: float
    threshold: float
    best_metric_value: float
    metric_name: str


class QEnsemble:
    """
    Learned Q-Ensemble combiner using Logistic Regression stacking.

    Trains a LR on the 3 anomaly scores → binary label, then tunes
    the threshold on predicted probabilities to maximize macro F-beta
    subject to a minimum benign recall constraint.
    """

    def __init__(self, n_models: int = 3, config: QEnsembleConfig | None = None):
        self.config = config or QEnsembleConfig()
        self.n_models = n_models
        self.lr: LogisticRegression | None = None
        self.threshold: float = 0.5
        self._optimized: bool = False

    def optimize(
        self,
        scores: list[np.ndarray],
        y_true: np.ndarray,
    ) -> EnsembleResult:
        """
        Train LR on scores, then tune threshold with benign recall constraint.

        Args:
            scores: List of anomaly score arrays, one per model. Shape (n_samples,).
            y_true: Binary labels (0=benign, 1=attack).

        Returns:
            EnsembleResult with LR coefficients and optimized threshold.
        """
        assert len(scores) == self.n_models

        X = np.stack(scores, axis=1)  # (n_samples, n_models)
        beta = self.config.beta
        min_benign_recall = self.config.min_benign_recall

        # ── 1. Train Logistic Regression ───────────────────────────────
        print("[Q-Ensemble] Training Logistic Regression stacking combiner...")
        self.lr = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        self.lr.fit(X, y_true)

        coefs = self.lr.coef_[0]
        intercept = self.lr.intercept_[0]
        print(f"[Q-Ensemble] LR coefficients: IF={coefs[0]:.4f}, "
              f"AE={coefs[1]:.4f}, SVM={coefs[2]:.4f}")
        print(f"[Q-Ensemble] LR intercept: {intercept:.4f}")

        # ── 2. Get predicted probabilities ─────────────────────────────
        probs = self.lr.predict_proba(X)[:, 1]  # P(attack)

        # ── 3. Threshold tuning with benign recall constraint ──────────
        thresholds = np.linspace(0.01, 0.99, 200)
        benign_mask = ~y_true.astype(bool)
        n_benign = benign_mask.sum()
        beta_sq = beta ** 2

        print(f"[Q-Ensemble] Tuning threshold (200 candidates, "
              f"min_benign_recall={min_benign_recall}, β={beta})...")

        best_score = -1.0
        best_threshold = 0.5

        for thr in thresholds:
            preds = (probs >= thr).astype(int)

            # Check benign recall constraint
            benign_correct = (preds[benign_mask] == 0).sum()
            benign_recall = benign_correct / max(n_benign, 1)
            if benign_recall < min_benign_recall:
                continue

            # Compute macro F-beta
            positives = y_true.astype(bool)
            # Attack class
            tp_a = (preds[positives] == 1).sum()
            fp_a = (preds[~positives] == 1).sum()
            fn_a = (preds[positives] == 0).sum()
            p_a = tp_a / max(tp_a + fp_a, 1)
            r_a = tp_a / max(tp_a + fn_a, 1)
            d_a = beta_sq * p_a + r_a
            fb_a = (1 + beta_sq) * p_a * r_a / d_a if d_a > 0 else 0

            # Benign class
            tp_b = benign_correct
            fp_b = fn_a  # attacks predicted as benign
            fn_b = fp_a  # benign predicted as attack
            p_b = tp_b / max(tp_b + fp_b, 1)
            r_b = tp_b / max(tp_b + fn_b, 1)
            d_b = beta_sq * p_b + r_b
            fb_b = (1 + beta_sq) * p_b * r_b / d_b if d_b > 0 else 0

            macro_fb = (fb_a + fb_b) / 2.0

            if macro_fb > best_score:
                best_score = macro_fb
                best_threshold = float(thr)

        self.threshold = best_threshold
        self._optimized = True

        print(f"[Q-Ensemble] Optimized threshold: {self.threshold:.4f}")
        print(f"[Q-Ensemble] Best macro F-beta (β={beta}): {best_score:.4f}")

        return EnsembleResult(
            coefficients=coefs.tolist(),
            intercept=float(intercept),
            threshold=self.threshold,
            best_metric_value=best_score,
            metric_name=f"macro_fbeta_b{beta}",
        )

    def combine_scores(self, scores: list[np.ndarray]) -> np.ndarray:
        """Return LR predicted probability of attack."""
        assert self.lr is not None, "Must call optimize() first"
        X = np.stack(scores, axis=1)
        return self.lr.predict_proba(X)[:, 1]

    def predict(self, scores: list[np.ndarray]) -> np.ndarray:
        """Binary prediction using LR probability and optimized threshold."""
        probs = self.combine_scores(scores)
        return (probs >= self.threshold).astype(int)

    def to_dict(self) -> dict:
        """Serialize ensemble config for export (Rust inference)."""
        coefs = self.lr.coef_[0].tolist() if self.lr else [0, 0, 0]
        intercept = float(self.lr.intercept_[0]) if self.lr else 0.0
        return {
            "type": "logistic_regression",
            "n_models": self.n_models,
            "coefficients": coefs,
            "intercept": intercept,
            "threshold": self.threshold,
            "model_names": ["isolation_forest", "autoencoder", "one_class_svm"],
            "optimized": self._optimized,
        }
