"""Anomaly detection models: Isolation Forest and Autoencoder.

Unsupervised anomaly detection to complement supervised failure prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

from src.constants import MODEL_AUTOENCODER, MODEL_ISOLATION_FOREST, TASK_ANOMALY_DETECTION
from src.models.base import BaseModel, SklearnModelWrapper


class IsolationForestDetector(SklearnModelWrapper):
    """Isolation Forest for unsupervised anomaly detection."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 200, "contamination": 0.05,
            "max_samples": "auto", "random_state": 42, "n_jobs": -1,
        }
        params.update(kwargs)
        super().__init__(
            MODEL_ISOLATION_FOREST, TASK_ANOMALY_DETECTION, IsolationForest(**params),
        )

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict anomalies: 1=normal, -1=anomaly (sklearn convention)."""
        return self.estimator.predict(x)

    def anomaly_scores(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous)."""
        return self.estimator.decision_function(x)


class _AutoencoderNetwork(nn.Module):
    """Symmetric autoencoder network."""

    def __init__(self, input_dim: int, encoding_dim: int = 16) -> None:
        super().__init__()
        mid = (input_dim + encoding_dim) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(mid, encoding_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, mid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseModel):
    """Autoencoder-based anomaly detector using reconstruction error."""

    def __init__(self, encoding_dim: int = 16, epochs: int = 50) -> None:
        super().__init__(MODEL_AUTOENCODER, TASK_ANOMALY_DETECTION)
        self._encoding_dim = encoding_dim
        self._epochs = epochs
        self._network: _AutoencoderNetwork | None = None
        self._threshold: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self, x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | None = None,
    ) -> dict[str, float]:
        """Train autoencoder on normal data to learn reconstruction."""
        x_arr = np.asarray(x_train, dtype=np.float32)
        self._network = _AutoencoderNetwork(x_arr.shape[1], self._encoding_dim).to(self.device)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        dataset = TensorDataset(torch.from_numpy(x_arr))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self._network.train()
        final_loss = 0.0
        for epoch in range(self._epochs):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self._network(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total += loss.item()
            final_loss = total / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info("AE Epoch {}/{} — loss: {:.6f}", epoch + 1, self._epochs, final_loss)

        # Set threshold as 95th percentile of training reconstruction error
        errors = self._reconstruction_errors(x_arr)
        self._threshold = float(np.percentile(errors, 95))
        self._is_trained = True
        self.metadata.training_samples = len(x_arr)
        logger.info("Autoencoder threshold set to {:.6f}", self._threshold)
        return {"final_loss": final_loss, "threshold": self._threshold}

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict anomalies: 1=anomaly, 0=normal."""
        errors = self._reconstruction_errors(np.asarray(x, dtype=np.float32))
        return (errors > self._threshold).astype(int)

    def anomaly_scores(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return reconstruction errors as anomaly scores."""
        return self._reconstruction_errors(np.asarray(x, dtype=np.float32))

    def _reconstruction_errors(self, x: np.ndarray) -> np.ndarray:
        if self._network is None:
            msg = "Model not trained"
            raise RuntimeError(msg)
        self._network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x).to(self.device)
            recon = self._network(tensor).cpu().numpy()
        return np.mean((x - recon) ** 2, axis=1)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{self.model_name}.pt"
        torch.save({
            "state_dict": self._network.state_dict() if self._network else None,
            "threshold": self._threshold, "encoding_dim": self._encoding_dim,
        }, model_path)
        self._save_metadata(path)
        return model_path

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(
            path / f"{self.model_name}.pt", map_location=self.device, weights_only=False,
        )
        self._threshold = checkpoint["threshold"]
        self._encoding_dim = checkpoint["encoding_dim"]
        self._load_metadata(path)
        self._is_trained = True
