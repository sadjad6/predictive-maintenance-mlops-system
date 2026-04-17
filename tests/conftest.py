"""Shared test fixtures for the predictive maintenance test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig, DataConfig, ModelConfig
from src.constants import (
    COL_CYCLE,
    COL_ENGINE_ID,
    COL_FAILURE_LABEL,
    COL_RUL,
    OPERATIONAL_SETTING_1,
    OPERATIONAL_SETTING_2,
    OPERATIONAL_SETTING_3,
    SENSOR_CURRENT,
    SENSOR_PRESSURE,
    SENSOR_ROTATION_SPEED,
    SENSOR_TEMPERATURE,
    SENSOR_VIBRATION,
    SENSOR_VOLTAGE,
)


@pytest.fixture
def small_data_config() -> DataConfig:
    """Small data config for fast tests."""
    return DataConfig(num_engines=5, min_cycles=50, max_cycles=80, failure_window=15)


@pytest.fixture
def small_model_config(tmp_path) -> ModelConfig:
    """Small model config for fast tests."""
    return ModelConfig(
        n_cv_splits=2, n_optuna_trials=2, lstm_epochs=3,
        lstm_sequence_length=5, models_dir=tmp_path / "models",
    )


@pytest.fixture
def test_config(small_data_config, small_model_config) -> AppConfig:
    """Complete test configuration."""
    return AppConfig(data=small_data_config, model=small_model_config)


@pytest.fixture
def sample_sensor_df() -> pd.DataFrame:
    """Create a small sample sensor DataFrame for testing."""
    rng = np.random.default_rng(42)
    n_engines = 3
    cycles_per_engine = 50
    rows = []

    for eid in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            rows.append({
                COL_ENGINE_ID: eid,
                COL_CYCLE: cycle,
                SENSOR_TEMPERATURE: 520 + cycle * 0.1 + rng.normal(0, 2),
                SENSOR_VIBRATION: 0.02 + cycle * 0.0002 + rng.normal(0, 0.003),
                SENSOR_PRESSURE: 14.7 - cycle * 0.01 + rng.normal(0, 0.2),
                SENSOR_ROTATION_SPEED: 9000 - cycle * 2 + rng.normal(0, 30),
                SENSOR_VOLTAGE: 230 - cycle * 0.05 + rng.normal(0, 1),
                SENSOR_CURRENT: 15 + cycle * 0.02 + rng.normal(0, 0.3),
                OPERATIONAL_SETTING_1: rng.uniform(-0.01, 0.01),
                OPERATIONAL_SETTING_2: rng.uniform(-0.005, 0.005),
                OPERATIONAL_SETTING_3: 100 + rng.uniform(-0.5, 0.5),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def labeled_sensor_df(sample_sensor_df) -> pd.DataFrame:
    """Sample sensor data with RUL and failure labels."""
    df = sample_sensor_df.copy()
    max_cycles = df.groupby(COL_ENGINE_ID)[COL_CYCLE].transform("max")
    df[COL_RUL] = max_cycles - df[COL_CYCLE]
    df[COL_FAILURE_LABEL] = (df[COL_RUL] <= 15).astype(int)
    return df


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample y_true, y_pred, y_proba for metric testing."""
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    y_proba = rng.uniform(0, 1, size=len(y_true))
    # Make proba correlate with true labels
    y_proba[y_true == 1] = np.clip(y_proba[y_true == 1] + 0.3, 0, 1)
    y_proba[y_true == 0] = np.clip(y_proba[y_true == 0] - 0.2, 0, 1)
    return y_true, y_pred, y_proba
