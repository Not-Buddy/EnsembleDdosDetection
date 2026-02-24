data links :- 

cicddos :- https://www.kaggle.com/datasets/dhoogla/cicddos2019


ensemble_ddos_detection/
├── __init__.py
├── config.py                    # Hyperparams & paths
├── data/
│   ├── __init__.py
│   ├── loader.py                # Load & merge parquets, binarize labels
│   └── preprocessor.py          # Clean, normalize, train/val/test split
├── models/
│   ├── __init__.py
│   ├── isolation_forest.py      # Scikit-learn Isolation Forest wrapper
│   ├── autoencoder.py           # PyTorch Autoencoder
│   ├── one_class_svm.py         # Scikit-learn One-Class SVM wrapper
│   └── q_ensemble.py            # Weighted score-level ensemble
├── training/
│   ├── __init__.py
│   └── trainer.py               # Orchestrates training of all 3 models
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               # Precision, Recall, F1, ROC-AUC, confusion matrix
└── export/
    ├── __init__.py
    └── exporter.py              # Export all models to ONNX
train.py                          # CLI entry-point for full pipeline




