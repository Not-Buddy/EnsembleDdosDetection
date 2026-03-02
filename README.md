# Ensemble DDoS Detection

Real-time DDoS mitigation using a Q-Ensemble of three one-class anomaly detectors (Isolation Forest, VAE Autoencoder, One-Class SVM) trained on the CIC-DDoS2019 dataset, with a Rust-based real-time packet ingestion agent for live detection.

## Quick Start

### Training (Python)

```bash
# 1. Install dependencies
uv sync

# 2. Train all models + evaluate
uv run python train.py

# 3. Export to ONNX (skip retraining)
uv run python train.py --export-only
```

> **Note:** The dataset must be placed in `Datasets/cicddos2019/` as parquet files.
> Download from: https://www.kaggle.com/datasets/dhoogla/cicddos2019

### Real-Time Detection (Rust)

```bash
# 1. Build the detection agent
cargo build --release

# 2. Run on a network interface (requires root/CAP_NET_RAW)
sudo ./target/release/ensemble-ddos-detection \
    --interface eth0 \
    --models-dir models/exported/onnx/ \
    --timeout 120
```

#### CLI Options

| Flag | Default | Description |
|---|---|---|
| `-i, --interface` | *(required)* | Network interface to capture on (e.g. `eth0`, `wlan0`) |
| `-m, --models-dir` | `models/exported/onnx` | Path to ONNX models + JSON configs |
| `-t, --timeout` | `120` | Flow inactivity timeout in seconds |
| `-s, --sweep-interval` | `10` | How often to classify expired flows (seconds) |

The agent captures live traffic, groups packets into bidirectional flows, computes 54 CICFlowMeter-style features per flow, and runs the ensemble through ONNX Runtime. Detected attacks are logged with 🚨 alerts showing source/destination IPs, protocol, and the combined detection probability.


## Model Performance

| Metric | Score |
|---|---|
| **F1 Score** | 0.995 |
| **Accuracy** | 99.0% |
| **Benign Recall** | 92.3% |
| **Attack Recall** | 99.6% |
| **ROC-AUC** | 0.996 |

### ROC Curve

![ROC Curve](outputs/roc_curve.png)

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)


## Architecture

### ML Processing Pipeline

```mermaid
%%{init: {"theme": "dark", "themeVariables": { "primaryColor": "#1a1a2e", "edgeLabelBackground":"#16213e", "tertiaryColor": "#1a1a2e"}}}%%
flowchart TB

    %% -------------------- 1. DATA --------------------
    subgraph DATA["Data Loading"]
        A1["CIC-DDoS2019<br/>Parquet files"]
        A2["Concatenate & binarize labels"]
        A3["Drop constant columns<br/>Keep numeric only"]
        A1 --> A2 --> A3
    end

    %% -------------------- 2. PREPROCESS --------------------
    subgraph PREPROCESS["Preprocessing"]
        B1["Log transform<br/>sign(x) · log1p(|x|)"]
        B2["Remove zero-variance columns"]
        B3["Inf → NaN<br/>Median imputation"]
        B4["Mutual information selection<br/>Drop bottom 10%"]
        B5["One-class split<br/>Train: benign only"]
        B6["StandardScaler<br/>Fit on benign train"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end

    %% -------------------- 3. DETECTORS --------------------
    subgraph DETECTORS["Anomaly Detectors (Benign-Trained)"]
        direction LR
        C1["Isolation Forest<br/>300 estimators"]
        C2["VAE Autoencoder<br/>Skip connections"]
        C3["One-Class SVM<br/>RBF kernel"]
    end

    %% -------------------- 4. ENSEMBLE --------------------
    subgraph ENSEMBLE["Q-Ensemble Stacking"]
        D1["Normalized anomaly scores<br/>(IF, VAE, SVM)"]
        D2["Logistic Regression<br/>Learned weights"]
        D3["Threshold tuning<br/>Max macro F-β"]
        D1 --> D2 --> D3
    end

    %% -------------------- 5. EVALUATION --------------------
    subgraph EVAL["Evaluation"]
        E1["Test metrics<br/>F1 · Accuracy · AUC"]
        E2["Per-attack detection rates"]
        E3["ROC curve & confusion matrix"]
        E1 --> E2 --> E3
    end

    %% -------------------- 6. EXPORT --------------------
    subgraph EXPORT["ONNX Export"]
        F1["Isolation Forest → ONNX"]
        F2["VAE → ONNX"]
        F3["SVM → ONNX"]
        F4["JSON configs<br/>Scaler + ensemble params"]
    end

    %% -------------------- FLOW --------------------
    DATA --> PREPROCESS
    PREPROCESS --> DETECTORS
    DETECTORS --> ENSEMBLE
    ENSEMBLE --> EVAL
    EVAL --> EXPORT

    %% -------------------- DARK THEME STYLING --------------------
    classDef default fill:#1a1a2e,stroke:#000,stroke-width:2px,color:#eee;
    classDef sub fill:#16213e,stroke:#000,stroke-width:2px,color:#fff;
    
    class DATA,PREPROCESS,DETECTORS,ENSEMBLE,EVAL,EXPORT sub;
```

### Real-Time Detection Agent

```mermaid
graph TB

    %% -------------------- 1. PACKET ACQUISITION --------------------
    subgraph L1["Packet Acquisition Layer"]
        A["NIC capture<br/>(pnet — TCP/UDP)"]
    end

    %% -------------------- 2. FLOW PROCESSING --------------------
    subgraph L2["Flow Processing Layer"]
        B["Flow aggregation<br/>(Bidirectional table — DashMap)"]
        C["Feature extraction<br/>(54 CICFlowMeter)"]
    end

    %% -------------------- 3. PREPROCESSING --------------------
    subgraph L3["Preprocessing Layer"]
        D["Log transform"]
        E["Standard scaler"]
    end

    %% -------------------- 4. MODEL INFERENCE --------------------
    subgraph L4["Model Inference Layer (ONNX Runtime)"]
        direction LR
        F["Isolation Forest"]
        G["Variational Autoencoder"]
        H["One-Class SVM"]
    end

    %% -------------------- 5. ENSEMBLE --------------------
    subgraph L5["Ensemble & Decision Layer"]
        I["Logistic Regression combiner"]
        J["Thresholding"]
        K{"Attack?"}
    end

    %% -------------------- 6. OUTPUT --------------------
    subgraph L6["Output Layer"]
        L["Alert<br/>(src/dst IPs, protocol, score)"]
        M["Benign traffic"]
    end

    %% -------------------- FLOW --------------------
    A --> B
    B --> C
    C --> D
    D --> E

    E --> F
    E --> G
    E --> H

    F --> I
    G --> I
    H --> I

    I --> J
    J --> K
    K -- "Yes" --> L
    K -- "No" --> M

    %% -------------------- STYLING (High Contrast / Professional) --------------------
    style L1 stroke:#000,stroke-width:2px
    style L2 stroke:#000,stroke-width:2px
    style L3 stroke:#000,stroke-width:2px
    style L4 stroke:#000,stroke-width:2px
    style L5 stroke:#000,stroke-width:2px
    style L6 stroke:#000,stroke-width:2px

    style A stroke:#000,stroke-width:2px
    style B stroke:#000,stroke-width:2px
    style C stroke:#000,stroke-width:2px
    style D stroke:#000,stroke-width:2px
    style E stroke:#000,stroke-width:2px
    style F stroke:#000,stroke-width:2px
    style G stroke:#000,stroke-width:2px
    style H stroke:#000,stroke-width:2px
    style I stroke:#000,stroke-width:2px
    style J stroke:#000,stroke-width:2px
    style K stroke:#000,stroke-width:2px
    style L stroke:#000,stroke-width:2px
    style M stroke:#000,stroke-width:2px
```

## Project Structure

```
ensemble_ddos_detection/          # Python training pipeline
├── config.py                     # Hyperparams & paths
├── data/
│   ├── loader.py                 # Load parquets, preserve attack types
│   └── preprocessor.py           # Log-transform, MI selection, scaling
├── models/
│   ├── isolation_forest.py       # Isolation Forest wrapper
│   ├── autoencoder.py            # VAE with skip connections
│   ├── one_class_svm.py          # One-Class SVM wrapper
│   └── q_ensemble.py             # Logistic Regression stacking combiner
├── training/
│   └── trainer.py                # Full pipeline orchestrator
├── evaluation/
│   └── metrics.py                # Metrics + per-attack-type evaluation
└── export/
    └── exporter.py               # Export all models to ONNX

src/                              # Rust real-time agent
├── main.rs                       # CLI, capture thread, sweep loop
├── capture.rs                    # Packet capture via pnet (TCP/UDP)
├── flow.rs                       # Concurrent flow table (DashMap)
├── features.rs                   # 54-feature CICFlowMeter extraction
├── preprocess.rs                 # Log-transform + StandardScaler
├── inference.rs                  # ONNX inference + LR combiner
├── config.rs                     # JSON config deserialization
├── tui/                          # Terminal UI (ratatui)
│   ├── app.rs                    # App state, event handling, tab navigation
│   ├── event.rs                  # Terminal event loop
│   ├── collectors/               # Background data collectors
│   │   ├── config.rs             # Config collector
│   │   ├── connections.rs        # Active connections collector
│   │   ├── geo.rs                # GeoIP lookups
│   │   ├── health.rs             # System health metrics
│   │   └── traffic.rs            # Traffic statistics
│   ├── platform/                 # OS-specific network interface detection
│   │   ├── linux.rs
│   │   ├── macos.rs
│   │   └── windows.rs
│   └── ui/                       # Tab UI renderers
│       ├── dashboard.rs          # Overview dashboard
│       ├── connections.rs        # Active connections table
│       ├── topology.rs           # Network topology view
│       ├── timeline.rs           # Traffic timeline graphs
│       ├── interfaces.rs         # Network interfaces tab
│       ├── ddos_logs.rs          # DDoS detection log viewer
│       ├── help.rs               # Help / keybindings tab
│       └── widgets.rs            # Shared widget helpers
└── test/                         # Unit tests
    ├── test_config.rs
    ├── test_features.rs
    ├── test_flow.rs
    ├── test_inference.rs
    └── test_preprocess.rs

models/exported/                  # Trained model artifacts
├── onnx/                         # ONNX models for Rust inference
├── *.pkl                         # Pickled sklearn models
├── *.pt                          # PyTorch checkpoints
└── *.json                        # Scaler, ensemble & normalization configs

train.py                          # Python CLI entry-point
```