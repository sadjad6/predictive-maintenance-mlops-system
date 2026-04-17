"""Base model interface and model registry.

Defines the abstract contract for all predictive models and
a registry for tracking trained model artifacts and metadata.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
from loguru import logger

from src.config import ModelConfig, get_config

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ModelMetadata:
    """Metadata for a trained model artifact."""

    model_name: str
    task_type: str
    trained_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    training_samples: int = 0
    version: str = "1.0.0"


class BaseModel(ABC):
    """Abstract base class for all predictive maintenance models.

    Subclasses must implement train, predict, save, and load methods.
    """

    def __init__(self, model_name: str, task_type: str) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.metadata = ModelMetadata(model_name=model_name, task_type=task_type)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(
        self,
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Train the model and return training metrics."""

    @abstractmethod
    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data."""

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray | None:
        """Generate probability predictions (classification only)."""
        return None

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Save model artifact to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model artifact from disk."""

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return self.metadata.hyperparameters

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names used during training."""
        self.metadata.feature_names = names

    def _save_metadata(self, model_dir: Path) -> None:
        """Save model metadata to JSON."""
        meta_path = model_dir / f"{self.model_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)

    def _load_metadata(self, model_dir: Path) -> None:
        """Load model metadata from JSON."""
        meta_path = model_dir / f"{self.model_name}_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            self.metadata = ModelMetadata(**data)


class ModelRegistry:
    """Tracks trained models, their metadata, and performance metrics.

    Provides model comparison and best-model selection functionality.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or get_config().model
        self._models: dict[str, BaseModel] = {}
        self._registry_path = Path(self.config.models_dir) / "registry.json"

    def register(self, model: BaseModel) -> None:
        """Register a trained model."""
        if not model.is_trained:
            msg = f"Cannot register untrained model: {model.model_name}"
            raise ValueError(msg)
        self._models[model.model_name] = model
        logger.info("Registered model: {}", model.model_name)

    def get_model(self, name: str) -> BaseModel:
        """Retrieve a registered model by name."""
        if name not in self._models:
            msg = f"Model not found: {name}"
            raise KeyError(msg)
        return self._models[name]

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def get_best_model(self, metric: str, higher_is_better: bool = True) -> BaseModel:
        """Select the best model based on a metric.

        Args:
            metric: Metric name to compare.
            higher_is_better: If True, highest value wins.

        Returns:
            The best performing model.
        """
        if not self._models:
            msg = "No models registered"
            raise ValueError(msg)

        scored_models = [
            (name, model.metadata.metrics.get(metric, float("-inf")))
            for name, model in self._models.items()
        ]

        best_name, best_score = (
            max(scored_models, key=lambda x: x[1])
            if higher_is_better
            else min(scored_models, key=lambda x: x[1])
        )

        logger.info(
            "Best model by {}: {} (score={:.4f})",
            metric,
            best_name,
            best_score,
        )
        return self._models[best_name]

    def save_registry(self) -> None:
        """Persist registry metadata to disk."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_data = {name: asdict(model.metadata) for name, model in self._models.items()}
        with open(self._registry_path, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)
        logger.info("Saved model registry to {}", self._registry_path)

    def comparison_table(self) -> pd.DataFrame:
        """Generate a model comparison table."""
        rows = []
        for name, model in self._models.items():
            row = {"model": name, "task": model.task_type, **model.metadata.metrics}
            rows.append(row)
        return pd.DataFrame(rows)


class SklearnModelWrapper(BaseModel):
    """Wrapper for scikit-learn compatible models."""

    def __init__(
        self,
        model_name: str,
        task_type: str,
        estimator: Any,
    ) -> None:
        super().__init__(model_name, task_type)
        self.estimator = estimator
        self.metadata.hyperparameters = _safe_get_params(estimator)

    def train(
        self,
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        self.estimator.fit(x_train, y_train)
        self._is_trained = True
        self.metadata.training_samples = len(x_train)
        logger.info("Trained {} on {} samples", self.model_name, len(x_train))
        return {}

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(x)  # type: ignore[no-any-return]

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray | None:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x)  # type: ignore[no-any-return]
        return None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{self.model_name}.joblib"
        joblib.dump(self.estimator, model_path)
        self._save_metadata(path)
        logger.info("Saved {} to {}", self.model_name, model_path)
        return model_path

    def load(self, path: str | Path) -> None:
        path = Path(path)
        model_path = path / f"{self.model_name}.joblib"
        self.estimator = joblib.load(model_path)
        self._load_metadata(path)
        self._is_trained = True
        logger.info("Loaded {} from {}", self.model_name, model_path)


def _safe_get_params(estimator: Any) -> dict[str, Any]:
    """Safely extract parameters from an estimator."""
    if hasattr(estimator, "get_params"):
        return estimator.get_params()  # type: ignore[no-any-return]
    return {}
