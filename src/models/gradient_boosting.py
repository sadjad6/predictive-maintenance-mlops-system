"""Gradient boosting models: XGBoost and LightGBM.

Advanced tree-based models for failure prediction and RUL estimation.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import xgboost as xgb

from src.constants import (
    MODEL_LIGHTGBM,
    MODEL_XGBOOST,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from src.models.base import SklearnModelWrapper


class XGBoostClassifierModel(SklearnModelWrapper):
    """XGBoost classifier for failure prediction."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "scale_pos_weight": 3.0, "eval_metric": "aucpr",
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        params.update(kwargs)
        super().__init__(MODEL_XGBOOST, TASK_CLASSIFICATION, xgb.XGBClassifier(**params))


class XGBoostRegressorModel(SklearnModelWrapper):
    """XGBoost regressor for RUL estimation."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "eval_metric": "rmse", "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        params.update(kwargs)
        super().__init__(f"{MODEL_XGBOOST}_regressor", TASK_REGRESSION, xgb.XGBRegressor(**params))


class LightGBMClassifierModel(SklearnModelWrapper):
    """LightGBM classifier for failure prediction."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 10,
            "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        params.update(kwargs)
        super().__init__(MODEL_LIGHTGBM, TASK_CLASSIFICATION, lgb.LGBMClassifier(**params))


class LightGBMRegressorModel(SklearnModelWrapper):
    """LightGBM regressor for RUL estimation."""

    def __init__(self, **kwargs: Any) -> None:
        params: dict[str, Any] = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 10,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        params.update(kwargs)
        super().__init__(f"{MODEL_LIGHTGBM}_regressor", TASK_REGRESSION, lgb.LGBMRegressor(**params))
