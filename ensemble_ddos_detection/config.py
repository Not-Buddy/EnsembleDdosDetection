"""
Central configuration for the Q-Ensemble DDoS Detection pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Datasets" / "cicddos2019"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models" / "exported"


# ── Columns to drop (constant / non-numeric / identifiers) ────────────
DROP_COLUMNS: list[str] = [
    # These are always 0 in CIC-DDoS2019
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "ECE Flag Count",
]

LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"


# ── Data split ratios ─────────────────────────────────────────────────
TRAIN_RATIO = 0.70   # benign-only for one-class training
VAL_RATIO = 0.15     # mixed: for threshold & weight tuning
TEST_RATIO = 0.15    # mixed: for final evaluation


# ── Feature engineering ────────────────────────────────────────────────
SKEW_THRESHOLD = 5.0         # features with |skew| > this get log-transformed
MI_DROP_PERCENTILE = 10.0    # bottom N% of mutual information features get dropped


@dataclass
class IsolationForestConfig:
    n_estimators: int = 300
    max_samples: int = 256       # lower = more aggressive splits, better separation
    contamination: str = "auto"  # let sklearn pick offset automatically
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class AutoencoderConfig:
    # Architecture
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.1
    use_skip_connections: bool = True

    # VAE
    kl_weight: float = 0.1       # max β for KL divergence
    kl_warmup_epochs: int = 10   # linearly ramp β from 0 to kl_weight

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-5
    scheduler: str = "cosine"   # "cosine" | "plateau" | "none"

    # Device
    device: str = "auto"


@dataclass
class OneClassSVMConfig:
    kernel: str = "rbf"
    gamma: str = "scale"
    nu: float = 0.01       # upper bound on fraction of training errors
    max_samples: int = 20_000  # subsample for scalability
    random_state: int = 42


@dataclass
class QEnsembleConfig:
    # Weight search grid resolution (steps per dimension)
    weight_grid_steps: int = 20
    # Metric to optimize
    optimize_metric: str = "macro_fbeta"
    # Beta for F-beta: <1 penalizes false positives more
    beta: float = 0.5
    # Hard constraint: reject any weight/threshold combo below this benign recall
    min_benign_recall: float = 0.85


@dataclass
class PipelineConfig:
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    one_class_svm: OneClassSVMConfig = field(default_factory=OneClassSVMConfig)
    q_ensemble: QEnsembleConfig = field(default_factory=QEnsembleConfig)

    dataset_dir: Path = DATASET_DIR
    output_dir: Path = OUTPUT_DIR
    models_dir: Path = MODELS_DIR

    random_state: int = 42
