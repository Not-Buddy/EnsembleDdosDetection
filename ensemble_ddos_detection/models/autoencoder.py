"""
PyTorch Autoencoder for one-class anomaly detection.

Anomaly score = reconstruction error (MSE per sample).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ensemble_ddos_detection.config import AutoencoderConfig


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class AutoencoderNetwork(nn.Module):
    """Symmetric encoder-decoder network."""

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float = 0.1):
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Decoder (mirror) ──────────────────────────────────────────
        decoder_layers: list[nn.Module] = []
        reversed_layers = list(reversed(hidden_layers))
        for i, h_dim in enumerate(reversed_layers[1:], 1):
            decoder_layers.extend([
                nn.Linear(reversed_layers[i - 1], h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        # Final reconstruction layer (no activation for continuous output)
        decoder_layers.append(nn.Linear(reversed_layers[-1] if reversed_layers else input_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderModel:
    """Autoencoder wrapper with training loop and anomaly scoring."""

    def __init__(self, input_dim: int, config: AutoencoderConfig | None = None):
        self.config = config or AutoencoderConfig()
        self.input_dim = input_dim
        self.device = _get_device(self.config.device)

        self.network = AutoencoderNetwork(
            input_dim=input_dim,
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = nn.MSELoss(reduction="none")

        # For score normalization
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutoencoderModel":
        """
        Train the autoencoder on benign-only data with early stopping.

        If X_val is provided, uses reconstruction error on benign validation
        samples for early stopping.
        """
        print(f"[Autoencoder] Training on {X_train.shape[0]:,} samples (device: {self.device})")
        print(f"[Autoencoder] Architecture: {self.input_dim} → {self.config.hidden_layers} → {self.input_dim}")

        train_ds = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Prepare benign validation set for early stopping
        val_benign_tensor = None
        if X_val is not None and y_val is not None:
            benign_val_mask = y_val == 0
            if benign_val_mask.any():
                val_benign_tensor = torch.FloatTensor(X_val[benign_val_mask]).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.max_epochs + 1):
            # ── Train ──────────────────────────────────────────────────
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                reconstructed = self.network(batch)
                loss = self.criterion(reconstructed, batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # ── Validate ───────────────────────────────────────────────
            if val_benign_tensor is not None:
                self.network.eval()
                with torch.no_grad():
                    val_recon = self.network(val_benign_tensor)
                    val_loss = self.criterion(val_recon, val_benign_tensor).mean().item()

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  Epoch {epoch:3d}/{self.config.max_epochs} — "
                        f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        print(f"  Early stopping at epoch {epoch} (patience={self.config.patience})")
                        break
            else:
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}/{self.config.max_epochs} — Train Loss: {avg_train_loss:.6f}")

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)

        # Calibrate score normalization on training data
        train_scores = self._raw_scores(X_train)
        self._score_min = float(train_scores.min())
        self._score_max = float(np.percentile(train_scores, 99.5))  # robustify
        print("[Autoencoder] Training complete.")
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        self.network.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False)
        scores_list: list[np.ndarray] = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.network(batch)
                mse = ((recon - batch) ** 2).mean(dim=1)
                scores_list.append(mse.cpu().numpy())

        return np.concatenate(scores_list)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores normalized to [0, 1]. Higher = more anomalous."""
        raw = self._raw_scores(X)
        denom = self._score_max - self._score_min
        if denom < 1e-10:
            return np.zeros(len(X))
        normalized = (raw - self._score_min) / denom
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 = attack, 0 = benign."""
        return (self.score(X) >= threshold).astype(int)
