"""Prefect tasks for pipeline steps.

Each task is an atomic, retryable unit of work with logging
and result caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from prefect import task

from src.config import AppConfig, get_config
from src.constants import COL_FAILURE_LABEL, COL_RUL, METRIC_ROC_AUC
from src.data.ingestion import DataIngestor
from src.data.simulator import SensorDataSimulator
from src.data.validation import DataValidator, ValidationReport
from src.features.engineering import FeatureEngineer
from src.features.labeling import FailureLabeler
from src.models.training import ModelTrainer

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@task(name="generate_sensor_data", retries=2, retry_delay_seconds=10)
def generate_sensor_data(config: AppConfig | None = None) -> pd.DataFrame:
    """Generate synthetic sensor data."""
    cfg = config or get_config()
    simulator = SensorDataSimulator(cfg.data)
    df = simulator.generate()
    simulator.save(df)
    logger.info("Generated {} sensor records", len(df))
    return df


@task(name="validate_data", retries=1)
def validate_data(df: pd.DataFrame, config: AppConfig | None = None) -> ValidationReport:
    """Validate sensor data quality."""
    cfg = config or get_config()
    validator = DataValidator(cfg.data)
    report = validator.validate(df)
    if not report.is_valid:
        logger.error("Data validation failed: {}", report.summary())
    return report


@task(name="ingest_data", retries=2, retry_delay_seconds=10)
def ingest_data(
    source_path: str | Path,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Ingest data from file."""
    cfg = config or get_config()
    ingestor = DataIngestor(cfg.data)
    return ingestor.ingest_from_file(source_path)


@task(name="engineer_features", retries=1)
def engineer_features(
    df: pd.DataFrame,
    config: AppConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run feature engineering pipeline."""
    cfg = config or get_config()
    engineer = FeatureEngineer(cfg.features)
    df_featured = engineer.transform(df)
    return df_featured, engineer.feature_columns


@task(name="add_labels", retries=1)
def add_labels(
    df: pd.DataFrame,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Add failure and RUL labels."""
    cfg = config or get_config()
    labeler = FailureLabeler(cfg.data)
    df = labeler.add_labels(df)
    df = labeler.clip_rul(df)
    return df


@task(name="train_classification_models", retries=1)
def train_classification_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: AppConfig | None = None,
) -> ModelTrainer:
    """Train all classification models."""
    from src.models.baseline import LogisticRegressionModel, RandomForestClassifierModel
    from src.models.gradient_boosting import LightGBMClassifierModel, XGBoostClassifierModel

    cfg = config or get_config()
    trainer = ModelTrainer(cfg.model)

    x = np.asarray(df[feature_cols])
    y = np.asarray(df[COL_FAILURE_LABEL])

    models = [
        LogisticRegressionModel(),
        RandomForestClassifierModel(),
        XGBoostClassifierModel(),
        LightGBMClassifierModel(),
    ]

    for model in models:
        try:
            trainer.train_and_evaluate(model, x, y, feature_cols)
        except Exception as e:
            logger.error("Failed to train {}: {}", model.model_name, e)

    return trainer


@task(name="train_regression_models", retries=1)
def train_regression_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: AppConfig | None = None,
) -> ModelTrainer:
    """Train all regression models for RUL."""
    from src.models.baseline import RandomForestRegressorModel
    from src.models.gradient_boosting import LightGBMRegressorModel, XGBoostRegressorModel

    cfg = config or get_config()
    trainer = ModelTrainer(cfg.model)

    x = np.asarray(df[feature_cols])
    y = np.asarray(df[COL_RUL])

    models = [
        RandomForestRegressorModel(),
        XGBoostRegressorModel(),
        LightGBMRegressorModel(),
    ]

    for model in models:
        try:
            trainer.train_and_evaluate(model, x, y, feature_cols)
        except Exception as e:
            logger.error("Failed to train {}: {}", model.model_name, e)

    return trainer


@task(name="select_best_model", retries=1)
def select_best_model(
    trainer: ModelTrainer,
    metric: str = METRIC_ROC_AUC,
    higher_is_better: bool = True,
) -> str:
    """Select the best model from the registry."""
    best = trainer.registry.get_best_model(metric, higher_is_better)
    logger.info("Selected best model: {}", best.model_name)
    return best.model_name


@task(name="save_models", retries=1)
def save_models(trainer: ModelTrainer) -> None:
    """Save all trained models to disk."""
    trainer.save_all_models()
