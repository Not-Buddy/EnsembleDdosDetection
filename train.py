#!/usr/bin/env python3
"""
Q-Ensemble DDoS Detection — Training Entry Point

Usage:
    uv run python train.py                 # Train all models
    uv run python train.py --export        # Train + export to ONNX
    uv run python train.py --export-only   # Export saved models (skip training)
"""

import argparse
import sys

from ensemble_ddos_detection.config import PipelineConfig
from ensemble_ddos_detection.training.trainer import train_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Q-Ensemble DDoS Detection models"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export trained models to ONNX after training",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training, only export existing models to ONNX",
    )
    parser.add_argument(
        "--svm-max-samples",
        type=int,
        default=20_000,
        help="Max samples for One-Class SVM training (default: 20000)",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=100,
        help="Max epochs for autoencoder training (default: 100)",
    )
    parser.add_argument(
        "--ae-patience",
        type=int,
        default=10,
        help="Early stopping patience for autoencoder (default: 10)",
    )
    args = parser.parse_args()

    # ── Build config ──────────────────────────────────────────────────
    config = PipelineConfig()
    config.one_class_svm.max_samples = args.svm_max_samples
    config.autoencoder.max_epochs = args.ae_epochs
    config.autoencoder.patience = args.ae_patience

    # ── Export-only mode ──────────────────────────────────────────────
    if args.export_only:
        from ensemble_ddos_detection.export.exporter import export_all
        export_all(config.models_dir)
        return 0

    # ── Train ─────────────────────────────────────────────────────────
    results = train_pipeline(config)

    # ── Export ─────────────────────────────────────────────────────────
    if args.export:
        from ensemble_ddos_detection.export.exporter import export_all
        export_all(config.models_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())

