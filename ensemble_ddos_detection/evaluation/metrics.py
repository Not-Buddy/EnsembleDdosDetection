"""
Evaluation metrics for anomaly detection.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true:   Ground truth (0 = benign, 1 = attack).
        y_pred:   Predicted labels.
        y_scores: Anomaly scores (for ROC-AUC). Optional.

    Returns:
        Dict with accuracy, precision, recall, f1, and optionally roc_auc.
    """
    results: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_scores is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            results["roc_auc"] = 0.0

    return results


def print_report(
    results: dict[str, float],
    title: str = "Evaluation Results",
) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key, value in results.items():
        print(f"  {key:<15s}: {value:.4f}")
    print(f"{'=' * 50}\n")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Classification Report",
) -> None:
    """Print sklearn classification report."""
    print(f"\n{title}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve — Q-Ensemble DDoS Detection",
) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Metrics] ROC curve saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix — Q-Ensemble",
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)

    labels = ["Benign", "Attack"]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], ",d"),
                ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Metrics] Confusion matrix saved to {save_path}")


def evaluate_per_attack_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: np.ndarray,
    title: str = "Per-Attack-Type Detection Rates",
) -> dict[str, dict[str, float]]:
    """
    Compute detection rate (recall) per attack type.

    Args:
        y_true: Binary ground truth (0=benign, 1=attack).
        y_pred: Binary predictions.
        attack_types: Original string labels (e.g. 'Syn', 'DrDoS_DNS', 'Benign').

    Returns:
        Dict mapping attack type → {count, detected, recall}.
    """
    unique_types = sorted(set(attack_types))
    results: dict[str, dict[str, float]] = {}

    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"  {'Attack Type':<20s} {'Count':>8s} {'Detected':>10s} {'Recall':>8s}")
    print(f"  {'-' * 55}")

    for atype in unique_types:
        mask = attack_types == atype
        count = int(mask.sum())
        if count == 0:
            continue

        true_sub = y_true[mask]
        pred_sub = y_pred[mask]

        if atype.lower() == "benign":
            # For benign: "detected" = correctly identified as benign (not flagged)
            detected = int((pred_sub == 0).sum())
            recall = detected / count
            label = "Benign (TN rate)"
        else:
            # For attacks: "detected" = correctly flagged as attack
            detected = int((pred_sub == 1).sum())
            recall = detected / count
            label = atype

        results[atype] = {"count": count, "detected": detected, "recall": recall}
        print(f"  {label:<20s} {count:>8,d} {detected:>10,d} {recall:>8.1%}")

    print(f"{'=' * 65}\n")
    return results

