"""
Variational Autoencoder (VAE) with skip connections for anomaly detection.

Anomaly score = reconstruction error + α * KL divergence.
Skip connections (U-Net style) preserve feature information through the bottleneck.
β-VAE warmup prevents posterior collapse.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ensemble_ddos_detection.config import AutoencoderConfig


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                print("[VAE] CUDA available but GPU incompatible, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


class VAENetwork(nn.Module):
    """
    Variational Autoencoder with optional skip connections.

    Encoder: input → h1 → h2 → (mu, log_var) → z (via reparameterization)
    Decoder: z (+skip h2) → h2' (+skip h1) → h1' → output

    Skip connections concatenate encoder activations to decoder inputs,
    preserving fine-grained feature information through the bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.1,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.input_dim = input_dim
        self.latent_dim = hidden_layers[-1]

        # ── Encoder layers (stored individually for skip connections) ───
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_layers[:-1]:  # all but last (latent)
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ))
            prev_dim = h_dim

        # ── VAE latent layer: mu and log_var ───────────────────────────
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, self.latent_dim)

        # ── Decoder layers ─────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList()
        reversed_hidden = list(reversed(hidden_layers[:-1]))  # mirror encoder

        # First decoder layer: latent_dim → first reversed hidden
        if reversed_hidden:
            dec_in = self.latent_dim + (self.latent_dim if use_skip else 0)
            # ^ skip from the pre-latent encoder output isn't available,
            #   but we skip from the encoder layer outputs
            # Actually: decoder input is just z (latent), first skip is from
            # the last encoder hidden layer
            first_in = self.latent_dim + (hidden_layers[-2] if use_skip else 0)
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(first_in, reversed_hidden[0]),
                nn.BatchNorm1d(reversed_hidden[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ))

            for i in range(1, len(reversed_hidden)):
                skip_dim = hidden_layers[-(i + 2)] if use_skip else 0
                self.decoder_layers.append(nn.Sequential(
                    nn.Linear(reversed_hidden[i - 1] + skip_dim, reversed_hidden[i]),
                    nn.BatchNorm1d(reversed_hidden[i]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ))

            last_dec_dim = reversed_hidden[-1]
        else:
            last_dec_dim = self.latent_dim

        # Final reconstruction layer
        self.fc_out = nn.Linear(last_dec_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encode input → (mu, log_var, encoder_hidden_outputs)."""
        hidden_outputs: list[torch.Tensor] = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            hidden_outputs.append(h)

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var, hidden_outputs

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def decode(self, z: torch.Tensor, hidden_outputs: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Decode z → reconstruction, using skip connections if available."""
        h = z
        for i, layer in enumerate(self.decoder_layers):
            if self.use_skip and hidden_outputs is not None:
                # Skip from corresponding encoder layer (reversed order)
                skip_idx = len(hidden_outputs) - 1 - i
                if skip_idx >= 0:
                    h = torch.cat([h, hidden_outputs[skip_idx]], dim=1)
            h = layer(h)

        return self.fc_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass. Returns (reconstruction, mu, log_var)."""
        mu, log_var, hidden_outputs = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, hidden_outputs)
        return recon, mu, log_var


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(mu, sigma) to N(0, 1), averaged over batch."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


class AutoencoderModel:
    """VAE wrapper with training loop, LR scheduling, and anomaly scoring."""

    def __init__(self, input_dim: int, config: AutoencoderConfig | None = None):
        self.config = config or AutoencoderConfig()
        self.input_dim = input_dim
        self.device = _get_device(self.config.device)

        self.network = VAENetwork(
            input_dim=input_dim,
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout,
            use_skip=self.config.use_skip_connections,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # LR Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

        self.mse_loss = nn.MSELoss(reduction="none")

        # Score normalization
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def _get_beta(self, epoch: int) -> float:
        """Linear β warmup for KL divergence."""
        if epoch <= self.config.kl_warmup_epochs:
            return self.config.kl_weight * (epoch / self.config.kl_warmup_epochs)
        return self.config.kl_weight

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutoencoderModel":
        """Train VAE on benign-only data with early stopping and LR scheduling."""
        print(f"[VAE] Training on {X_train.shape[0]:,} samples (device: {self.device})")
        print(f"[VAE] Architecture: {self.input_dim} → {self.config.hidden_layers} "
              f"(skip={self.config.use_skip_connections}, scheduler={self.config.scheduler})")

        train_ds = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Benign validation set for early stopping
        val_benign_tensor = None
        if X_val is not None and y_val is not None:
            benign_val_mask = y_val == 0
            if benign_val_mask.any():
                val_benign_tensor = torch.FloatTensor(X_val[benign_val_mask]).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.max_epochs + 1):
            beta = self._get_beta(epoch)

            # ── Train ──────────────────────────────────────────────────
            self.network.train()
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            n_batches = 0

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                recon, mu, log_var = self.network(batch)

                recon_loss = self.mse_loss(recon, batch).mean()
                kl_loss = kl_divergence(mu, log_var)
                loss = recon_loss + beta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                n_batches += 1

            avg_recon = epoch_recon_loss / n_batches
            avg_kl = epoch_kl_loss / n_batches

            # Step LR scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch)

            # ── Validate ───────────────────────────────────────────────
            if val_benign_tensor is not None:
                self.network.eval()
                with torch.no_grad():
                    val_recon, val_mu, val_log_var = self.network(val_benign_tensor)
                    val_recon_loss = self.mse_loss(val_recon, val_benign_tensor).mean().item()
                    val_kl = kl_divergence(val_mu, val_log_var).item()
                    val_loss = val_recon_loss + beta * val_kl

                if epoch % 10 == 0 or epoch == 1:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Epoch {epoch:3d}/{self.config.max_epochs} — "
                        f"ReconL: {avg_recon:.4f} KL: {avg_kl:.4f} β: {beta:.3f} "
                        f"| Val: {val_loss:.4f} LR: {lr:.2e}"
                    )

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)

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
                    print(f"  Epoch {epoch:3d}/{self.config.max_epochs} — "
                          f"ReconL: {avg_recon:.4f} KL: {avg_kl:.4f} β: {beta:.3f}")

        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)

        # Calibrate score normalization on training data
        train_scores = self._raw_scores(X_train)
        self._score_min = float(train_scores.min())
        self._score_max = float(np.percentile(train_scores, 99.5))
        print("[VAE] Training complete.")
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score per sample: MSE + α * KL divergence.

        Both components contribute: reconstruction error catches
        obvious anomalies, KL catches samples in unusual latent regions.
        """
        self.network.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False)
        scores_list: list[np.ndarray] = []

        alpha = self.config.kl_weight  # same weight as training

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, mu, log_var = self.network(batch)

                # Per-sample MSE
                mse = ((recon - batch) ** 2).mean(dim=1)

                # Per-sample KL divergence
                kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)

                score = mse + alpha * kl
                scores_list.append(score.cpu().numpy())

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
