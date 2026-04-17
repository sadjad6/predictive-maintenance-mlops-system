"""Prefect flows for end-to-end pipeline orchestration.

Defines composable flows for data ingestion, feature engineering,
training, evaluation, and deployment.
"""

from __future__ import annotations

from loguru import logger
from prefect import flow

from src.config import get_config
from src.constants import METRIC_MAE, METRIC_ROC_AUC
from src.pipeline.tasks import (
    add_labels,
    engineer_features,
    generate_sensor_data,
    save_models,
    select_best_model,
    train_classification_models,
    train_regression_models,
    validate_data,
)


@flow(name="data_ingestion_flow", log_prints=True)
def data_ingestion_flow():
    """Generate and validate sensor data."""
    config = get_config()
    df = generate_sensor_data(config)
    report = validate_data(df, config)

    if not report.is_valid:
        logger.error("Data validation failed — aborting pipeline")
        raise ValueError(report.summary())

    logger.info("Data ingestion complete: {}", report.summary())
    return df


@flow(name="feature_engineering_flow", log_prints=True)
def feature_engineering_flow(df=None):
    """Run feature engineering and labeling."""
    config = get_config()

    if df is None:
        df = data_ingestion_flow()

    df = add_labels(df, config)
    df, feature_cols = engineer_features(df, config)

    logger.info(
        "Features: {} columns, {} rows",
        len(feature_cols),
        len(df),
    )
    return df, feature_cols


@flow(name="training_flow", log_prints=True)
def training_flow(df=None, feature_cols=None):
    """Train and evaluate all models."""
    config = get_config()

    if df is None or feature_cols is None:
        df, feature_cols = feature_engineering_flow()

    clf_trainer = train_classification_models(df, feature_cols, config)
    reg_trainer = train_regression_models(df, feature_cols, config)

    best_clf = select_best_model(clf_trainer, METRIC_ROC_AUC, True)
    best_reg = select_best_model(reg_trainer, METRIC_MAE, False)

    save_models(clf_trainer)
    save_models(reg_trainer)

    logger.info("Best classifier: {}, Best regressor: {}", best_clf, best_reg)
    return clf_trainer, reg_trainer


@flow(name="full_pipeline", log_prints=True)
def full_pipeline_flow():
    """End-to-end pipeline: data → features → training → save."""
    logger.info("Starting full ML pipeline")

    df = data_ingestion_flow()
    df, feature_cols = feature_engineering_flow(df)
    clf_trainer, reg_trainer = training_flow(df, feature_cols)

    logger.info("Full pipeline complete")
    return {
        "classification_results": clf_trainer.results,
        "regression_results": reg_trainer.results,
    }


if __name__ == "__main__":
    full_pipeline_flow()
