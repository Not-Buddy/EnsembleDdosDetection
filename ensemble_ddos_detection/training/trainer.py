"""
Training pipeline orchestrator.

Trains all three anomaly detectors and the Q-Ensemble combiner,
then evaluates and saves results.
"""

import time
import json
import pickle
import numpy as np
from pathlib import Path

from ensemble_ddos_detection.config import PipelineConfig
from ensemble_ddos_detection.data.loader import load_dataset
from ensemble_ddos_detection.data.preprocessor import preprocess
from ensemble_ddos_detection.models.isolation_forest import IsolationForestModel
from ensemble_ddos_detection.models.autoencoder import AutoencoderModel
from ensemble_ddos_detection.models.one_class_svm import OneClassSVMModel
from ensemble_ddos_detection.models.q_ensemble import QEnsemble
from ensemble_ddos_detection.evaluation.metrics import (
    evaluate,
    print_report,
    print_classification_report,
    plot_roc_curve,
    plot_confusion_matrix,
)


def train_pipeline(config: PipelineConfig | None = None) -> dict:
    """
    Full training pipeline:
        1. Load & preprocess data
        2. Train Isolation Forest
        3. Train Autoencoder
        4. Train One-Class SVM
        5. Score validation set → optimize Q-Ensemble
        6. Evaluate on test set
        7. Save models + artifacts

    Returns:
        Dict with final evaluation metrics.
    """
    config = config or PipelineConfig()
    start_time = time.time()

    # ── 1. Data ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)
    X, y = load_dataset(config.dataset_dir)
    splits = preprocess(X, y, random_state=config.random_state)

    # ── 2. Isolation Forest ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: Training Isolation Forest")
    print("=" * 60)
    if_model = IsolationForestModel(config.isolation_forest)
    if_model.fit(splits.X_train)

    # ── 3. Autoencoder ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: Training Autoencoder")
    print("=" * 60)
    ae_model = AutoencoderModel(splits.n_features, config.autoencoder)
    ae_model.fit(splits.X_train, splits.X_val, splits.y_val)

    # ── 4. One-Class SVM ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4: Training One-Class SVM")
    print("=" * 60)
    svm_model = OneClassSVMModel(config.one_class_svm)
    svm_model.fit(splits.X_train)

    # ── 5. Q-Ensemble Optimization ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 5: Q-Ensemble Weight Optimization")
    print("=" * 60)
    val_scores = [
        if_model.score(splits.X_val),
        ae_model.score(splits.X_val),
        svm_model.score(splits.X_val),
    ]

    ensemble = QEnsemble(n_models=3, config=config.q_ensemble)
    ensemble_result = ensemble.optimize(val_scores, splits.y_val)

    # ── Individual model evaluation on validation set ──────────────────
    print("\n--- Individual Model Validation Results ---")
    model_names = ["Isolation Forest", "Autoencoder", "One-Class SVM"]
    models = [if_model, ae_model, svm_model]
    for name, score_arr in zip(model_names, val_scores):
        pred = (score_arr >= ensemble.threshold).astype(int)
        results = evaluate(splits.y_val, pred, score_arr)
        print_report(results, title=f"{name} (Validation)")

    # ── 6. Test Set Evaluation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 6: Final Evaluation on Test Set")
    print("=" * 60)
    test_scores = [
        if_model.score(splits.X_test),
        ae_model.score(splits.X_test),
        svm_model.score(splits.X_test),
    ]

    ensemble_test_scores = ensemble.combine_scores(test_scores)
    ensemble_test_preds = ensemble.predict(test_scores)

    final_results = evaluate(splits.y_test, ensemble_test_preds, ensemble_test_scores)
    print_report(final_results, title="Q-Ensemble (Test Set)")
    print_classification_report(splits.y_test, ensemble_test_preds, title="Q-Ensemble Classification Report")

    # Individual test results
    print("--- Individual Model Test Results ---")
    for name, score_arr in zip(model_names, test_scores):
        pred = (score_arr >= ensemble.threshold).astype(int)
        results = evaluate(splits.y_test, pred, score_arr)
        print_report(results, title=f"{name} (Test)")

    # ── 7. Save Everything ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 7: Saving Models & Artifacts")
    print("=" * 60)

    models_dir = config.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sklearn models
    with open(models_dir / "isolation_forest.pkl", "wb") as f:
        pickle.dump(if_model, f)
    print(f"  Saved: {models_dir / 'isolation_forest.pkl'}")

    with open(models_dir / "one_class_svm.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    print(f"  Saved: {models_dir / 'one_class_svm.pkl'}")

    # Save autoencoder (PyTorch state dict + config)
    import torch
    torch.save({
        "state_dict": ae_model.network.state_dict(),
        "input_dim": ae_model.input_dim,
        "config": ae_model.config,
        "score_min": ae_model._score_min,
        "score_max": ae_model._score_max,
    }, models_dir / "autoencoder.pt")
    print(f"  Saved: {models_dir / 'autoencoder.pt'}")

    # Save scaler parameters as JSON (for Rust)
    scaler_params = {
        "mean": splits.scaler.mean_.tolist(),
        "scale": splits.scaler.scale_.tolist(),
        "feature_names": splits.feature_names,
        "n_features": splits.n_features,
    }
    with open(models_dir / "scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Saved: {models_dir / 'scaler.json'}")

    # Save ensemble config
    ensemble_config = ensemble.to_dict()
    ensemble_config["ensemble_result"] = {
        "weights": ensemble_result.weights,
        "threshold": ensemble_result.threshold,
        "best_metric_value": ensemble_result.best_metric_value,
        "metric_name": ensemble_result.metric_name,
    }
    with open(models_dir / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"  Saved: {models_dir / 'ensemble_config.json'}")

    # Save scaler (pickle for Python reuse)
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(splits.scaler, f)

    # Save normalization bounds for IF and SVM
    norm_params = {
        "isolation_forest": {
            "p1": if_model._p1,
            "p99": if_model._p99,
        },
        "autoencoder": {
            "score_min": ae_model._score_min,
            "score_max": ae_model._score_max,
        },
        "one_class_svm": {
            "score_min": svm_model._score_min,
            "score_max": svm_model._score_max,
        },
    }
    with open(models_dir / "normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"  Saved: {models_dir / 'normalization_params.json'}")

    # Plots
    plot_roc_curve(
        splits.y_test, ensemble_test_scores,
        save_path=output_dir / "roc_curve.png",
    )
    plot_confusion_matrix(
        splits.y_test, ensemble_test_preds,
        save_path=output_dir / "confusion_matrix.png",
    )

    # Save final metrics as JSON
    final_results["training_time_seconds"] = time.time() - start_time
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"  Saved: {output_dir / 'test_metrics.json'}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return final_results
