"""Model training orchestrator with cross-validation and hyperparameter tuning.

Provides time-series-aware CV splits, Optuna-based hyperparameter search,
and a unified training interface for all model types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from src.config import ModelConfig, get_config
from src.constants import (
    METRIC_F1,
    METRIC_MAE,
    METRIC_RMSE,
    METRIC_ROC_AUC,
    TASK_CLASSIFICATION,
)
from src.models.base import BaseModel, ModelRegistry


@dataclass
class TrainingResult:
    """Result of a model training run."""

    model_name: str
    task_type: str
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    mean_scores: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Orchestrates model training with CV and hyperparameter tuning.

    Supports time-series aware cross-validation to prevent data leakage.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or get_config().model
        self.registry = ModelRegistry(self.config)
        self._results: list[TrainingResult] = []

    @property
    def results(self) -> list[TrainingResult]:
        return list(self._results)

    def train_and_evaluate(
        self,
        model: BaseModel,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """Train model with time-series cross-validation.

        Args:
            model: Model instance to train.
            x: Feature matrix.
            y: Target array.
            feature_names: Optional list of feature names.

        Returns:
            TrainingResult with CV scores.
        """
        logger.info("Training {} with {}-fold CV", model.model_name, self.config.n_cv_splits)

        if feature_names:
            model.set_feature_names(feature_names)

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        cv_scores = self._cross_validate(model, x_arr, y_arr)

        # Final training on all data
        model.train(x_arr, y_arr)
        model.metadata.metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}

        self.registry.register(model)

        result = TrainingResult(
            model_name=model.model_name,
            task_type=model.task_type,
            cv_scores=cv_scores,
            mean_scores=model.metadata.metrics,
        )
        self._results.append(result)

        logger.info(
            "{} CV results: {}",
            model.model_name,
            {k: f"{np.mean(v):.4f}±{np.std(v):.4f}" for k, v in cv_scores.items()},
        )
        return result

    def _cross_validate(
        self, model: BaseModel, x: np.ndarray, y: np.ndarray,
    ) -> dict[str, list[float]]:
        """Run time-series aware cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.config.n_cv_splits)
        is_clf = model.task_type == TASK_CLASSIFICATION

        metric_keys = [METRIC_ROC_AUC, METRIC_F1] if is_clf else [METRIC_RMSE, METRIC_MAE]
        scores: dict[str, list[float]] = {k: [] for k in metric_keys}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create a fresh copy for each fold
            fold_model = _clone_model(model)
            fold_model.train(x_train, y_train)

            fold_scores = _compute_fold_metrics(fold_model, x_val, y_val, is_clf)
            for key in metric_keys:
                scores[key].append(fold_scores.get(key, 0.0))

            logger.debug("Fold {}: {}", fold + 1, fold_scores)

        return scores

    def save_all_models(self) -> None:
        """Save all registered models to disk."""
        models_dir = self.config.models_dir
        for name in self.registry.list_models():
            model = self.registry.get_model(name)
            model.save(models_dir / name)
        self.registry.save_registry()
        logger.info("Saved all models to {}", models_dir)


def _clone_model(model: BaseModel) -> BaseModel:
    """Create a fresh instance of the same model type."""
    return model.__class__()


def _compute_fold_metrics(
    model: BaseModel, x_val: np.ndarray, y_val: np.ndarray, is_classification: bool,
) -> dict[str, float]:
    """Compute metrics for a single CV fold."""
    from math import sqrt

    from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

    preds = model.predict(x_val)
    metrics: dict[str, float] = {}

    if is_classification:
        proba = model.predict_proba(x_val)
        if proba is not None and len(np.unique(y_val)) > 1:
            metrics[METRIC_ROC_AUC] = roc_auc_score(y_val, proba[:, 1])
        metrics[METRIC_F1] = f1_score(y_val, preds, zero_division=0)
    else:
        mse = float(np.mean((y_val - preds) ** 2))
        metrics[METRIC_RMSE] = sqrt(mse)
        metrics[METRIC_MAE] = mean_absolute_error(y_val, preds)

    return metrics
