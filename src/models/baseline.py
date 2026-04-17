"""Baseline models: Logistic Regression and Random Forest.

Provides simple but interpretable baselines for failure prediction.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.constants import (
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from src.models.base import SklearnModelWrapper


class LogisticRegressionModel(SklearnModelWrapper):
    """Logistic Regression baseline for failure classification."""

    def __init__(self, **kwargs: Any) -> None:
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            **kwargs,
        )
        super().__init__(
            model_name=MODEL_LOGISTIC_REGRESSION,
            task_type=TASK_CLASSIFICATION,
            estimator=estimator,
        )
        self._scaler = StandardScaler()

    def train(self, x_train: Any, y_train: Any) -> dict[str, float]:
        """Train with feature scaling (required for LogReg)."""
        x_scaled = self._scaler.fit_transform(x_train)
        return super().train(x_scaled, y_train)

    def predict(self, x: Any) -> Any:
        x_scaled = self._scaler.transform(x)
        return super().predict(x_scaled)

    def predict_proba(self, x: Any) -> Any:
        x_scaled = self._scaler.transform(x)
        return super().predict_proba(x_scaled)


class RandomForestClassifierModel(SklearnModelWrapper):
    """Random Forest for failure classification with feature importance."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        estimator = RandomForestClassifier(**params)
        super().__init__(
            model_name=MODEL_RANDOM_FOREST,
            task_type=TASK_CLASSIFICATION,
            estimator=estimator,
        )

    @property
    def feature_importances(self) -> Any:
        """Return feature importances from the trained model."""
        if not self.is_trained:
            return None
        return self.estimator.feature_importances_


class RandomForestRegressorModel(SklearnModelWrapper):
    """Random Forest for RUL regression."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        estimator = RandomForestRegressor(**params)
        super().__init__(
            model_name=f"{MODEL_RANDOM_FOREST}_regressor",
            task_type=TASK_REGRESSION,
            estimator=estimator,
        )
