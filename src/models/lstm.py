"""LSTM models for time-series failure prediction and RUL estimation.

PyTorch-based LSTM networks that process sequential sensor data
with early stopping and learning rate scheduling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig, get_config
from src.constants import MODEL_LSTM, TASK_CLASSIFICATION
from src.models.base import BaseModel

if TYPE_CHECKING:
    import pandas as pd


class _LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class LSTMModel(BaseModel):
    """LSTM model for time-series predictive maintenance tasks."""

    def __init__(
        self,
        task_type: str = TASK_CLASSIFICATION,
        config: ModelConfig | None = None,
    ) -> None:
        name = f"{MODEL_LSTM}_{task_type}"
        super().__init__(model_name=name, task_type=task_type)
        self.config = config or get_config().model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._network: _LSTMNetwork | None = None
        self._input_size: int = 0

    def train(
        self,
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Train the LSTM on sequential data.

        Args:
            x_train: 3D array of shape (samples, sequence_length, features).
            y_train: 1D array of targets.
        """
        x_arr = np.asarray(x_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32)

        if x_arr.ndim == 2:
            x_arr = self._create_sequences(x_arr)

        self._input_size = x_arr.shape[2]
        output_size = 1

        self._network = _LSTMNetwork(
            self._input_size,
            self.config.lstm_hidden_size,
            self.config.lstm_num_layers,
            output_size,
        ).to(self.device)

        loss_fn = nn.BCEWithLogitsLoss() if self.task_type == TASK_CLASSIFICATION else nn.MSELoss()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.config.lstm_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        dataset = TensorDataset(
            torch.from_numpy(x_arr),
            torch.from_numpy(y_arr.reshape(-1, 1)),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.lstm_batch_size,
            shuffle=True,
        )

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 10

        self._network.train()
        for epoch in range(self.config.lstm_epochs):
            epoch_loss = self._train_epoch(loader, loss_fn, optimizer)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch {}", epoch + 1)
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch {}/{} — loss: {:.6f}", epoch + 1, self.config.lstm_epochs, epoch_loss
                )

        self._is_trained = True
        self.metadata.training_samples = len(x_arr)
        return {"final_loss": best_loss}

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate predictions from sequential input."""
        if self._network is None:
            msg = "Model not trained"
            raise RuntimeError(msg)

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 2:
            x_arr = self._create_sequences(x_arr)

        self._network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_arr).to(self.device)
            output = self._network(tensor).cpu().numpy().flatten()

        if self.task_type == TASK_CLASSIFICATION:
            return (torch.sigmoid(torch.from_numpy(output)).numpy() > 0.5).astype(int)
        return output

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray | None:
        """Return probability predictions for classification."""
        if self.task_type != TASK_CLASSIFICATION or self._network is None:
            return None

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 2:
            x_arr = self._create_sequences(x_arr)

        self._network.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_arr).to(self.device)
            logits = self._network(tensor).cpu().numpy().flatten()
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            return np.column_stack([1 - probs, probs])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{self.model_name}.pt"
        torch.save(
            {
                "state_dict": self._network.state_dict() if self._network else None,
                "input_size": self._input_size,
                "config": {
                    "hidden_size": self.config.lstm_hidden_size,
                    "num_layers": self.config.lstm_num_layers,
                },
            },
            model_path,
        )
        self._save_metadata(path)
        return model_path

    def load(self, path: str | Path) -> None:
        path = Path(path)
        model_path = path / f"{self.model_name}.pt"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self._input_size = checkpoint["input_size"]
        cfg = checkpoint["config"]
        self._network = _LSTMNetwork(
            self._input_size,
            cfg["hidden_size"],
            cfg["num_layers"],
            1,
        ).to(self.device)
        self._network.load_state_dict(checkpoint["state_dict"])
        self._load_metadata(path)
        self._is_trained = True

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: Any,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            optimizer.zero_grad()
            output = self._network(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _create_sequences(self, x: np.ndarray) -> np.ndarray:
        seq_len = min(self.config.lstm_sequence_length, len(x))
        sequences = []
        for i in range(len(x) - seq_len + 1):
            sequences.append(x[i : i + seq_len])
        return np.array(sequences)
