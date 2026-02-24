"""
Export trained models to ONNX format for Rust inference.

All three models + scaler → ONNX + JSON config files.
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path

from ensemble_ddos_detection.config import MODELS_DIR
from ensemble_ddos_detection.models.autoencoder import AutoencoderModel, AutoencoderNetwork


def export_autoencoder_onnx(
    model_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """Export PyTorch autoencoder to ONNX."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_dim = checkpoint["input_dim"]
    config = checkpoint["config"]

    network = AutoencoderNetwork(
        input_dim=input_dim,
        hidden_layers=config.hidden_layers,
        dropout=0.0,  # no dropout for inference
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    dummy_input = torch.randn(1, input_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        network,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,  # use legacy TorchScript exporter (avoids onnxscript converter bugs)
    )
    print(f"[Export] Autoencoder → {output_path}")


def export_sklearn_onnx(
    model_path: Path,
    output_path: Path,
    n_features: int,
) -> None:
    """Export sklearn model (IF or OC-SVM) to ONNX via skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[Export] WARNING: skl2onnx not installed. Skipping sklearn ONNX export.")
        print("  Install via: uv add skl2onnx")
        return

    with open(model_path, "rb") as f:
        wrapper = pickle.load(f)

    # Access the underlying sklearn model
    sklearn_model = wrapper.sklearn_model

    initial_type = [("input", FloatTensorType([None, n_features]))]
    # Pin ai.onnx.ml to v3 to avoid skl2onnx "version 4 not supported" error
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=initial_type,
        target_opset={"": 17, "ai.onnx.ml": 3},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[Export] {model_path.stem} → {output_path}")


def export_all(models_dir: Path = MODELS_DIR) -> None:
    """
    Export all trained models to ONNX.

    Expects the following files in models_dir:
      - autoencoder.pt
      - isolation_forest.pkl
      - one_class_svm.pkl
      - scaler.json (for n_features)
    """
    print("\n" + "=" * 60)
    print("  Exporting Models to ONNX")
    print("=" * 60)

    onnx_dir = models_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Read n_features from scaler config
    with open(models_dir / "scaler.json") as f:
        scaler_config = json.load(f)
    n_features = scaler_config["n_features"]

    # Autoencoder
    ae_path = models_dir / "autoencoder.pt"
    if ae_path.exists():
        export_autoencoder_onnx(ae_path, onnx_dir / "autoencoder.onnx")
    else:
        print(f"[Export] WARNING: {ae_path} not found, skipping autoencoder export")

    # Isolation Forest
    if_path = models_dir / "isolation_forest.pkl"
    if if_path.exists():
        export_sklearn_onnx(if_path, onnx_dir / "isolation_forest.onnx", n_features)
    else:
        print(f"[Export] WARNING: {if_path} not found, skipping IF export")

    # One-Class SVM
    svm_path = models_dir / "one_class_svm.pkl"
    if svm_path.exists():
        export_sklearn_onnx(svm_path, onnx_dir / "one_class_svm.onnx", n_features)
    else:
        print(f"[Export] WARNING: {svm_path} not found, skipping SVM export")

    # Copy supporting configs to onnx dir
    for json_file in ["scaler.json", "ensemble_config.json", "normalization_params.json"]:
        src = models_dir / json_file
        dst = onnx_dir / json_file
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"[Export] Copied {json_file} → {onnx_dir}")

    print(f"\n[Export] All exports complete → {onnx_dir}")


if __name__ == "__main__":
    export_all()
