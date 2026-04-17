"""Tests for feature engineering module."""

from __future__ import annotations

import pandas as pd

from src.config import FeatureConfig
from src.constants import COL_CYCLE, COL_ENGINE_ID, SENSOR_TEMPERATURE
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_transform_adds_features(self, sample_sensor_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer()
        result = engineer.transform(sample_sensor_df)
        assert len(result.columns) > len(sample_sensor_df.columns)
        assert len(engineer.feature_columns) > 0

    def test_rolling_features_created(self, sample_sensor_df: pd.DataFrame) -> None:
        config = FeatureConfig(rolling_windows=(5,), lag_periods=(1,))
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_sensor_df)
        assert f"{SENSOR_TEMPERATURE}_roll_mean_5" in result.columns
        assert f"{SENSOR_TEMPERATURE}_roll_std_5" in result.columns

    def test_lag_features_created(self, sample_sensor_df: pd.DataFrame) -> None:
        config = FeatureConfig(rolling_windows=(5,), lag_periods=(1, 3))
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_sensor_df)
        assert f"{SENSOR_TEMPERATURE}_lag_1" in result.columns
        assert f"{SENSOR_TEMPERATURE}_lag_3" in result.columns

    def test_no_data_leakage_in_rolling(self, sample_sensor_df: pd.DataFrame) -> None:
        """Rolling features should not use future values."""
        config = FeatureConfig(rolling_windows=(5,), lag_periods=())
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_sensor_df)

        # For engine 1, the rolling mean at cycle 5 should only use cycles 1-5
        engine1 = result[result[COL_ENGINE_ID] == 1].sort_values(COL_CYCLE)
        raw_values = sample_sensor_df[
            sample_sensor_df[COL_ENGINE_ID] == 1
        ].sort_values(COL_CYCLE)[SENSOR_TEMPERATURE].iloc[:5]

        roll_col = f"{SENSOR_TEMPERATURE}_roll_mean_5"
        if roll_col in engine1.columns:
            # Find the row corresponding to cycle 5 (iloc would be wrong if cycle 1 was dropped)
            cycle_5_row = engine1[engine1[COL_CYCLE] == 5]
            if not cycle_5_row.empty:
                computed_mean = cycle_5_row.iloc[0][roll_col]
                expected_mean = raw_values.mean()
                assert abs(computed_mean - expected_mean) < 0.01

    def test_no_nan_in_output(self, sample_sensor_df: pd.DataFrame) -> None:
        """Output should have no NaN values after dropping."""
        config = FeatureConfig(rolling_windows=(3,), lag_periods=(1,))
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_sensor_df)
        assert result.isna().sum().sum() == 0

    def test_cycle_normalized_feature(self, sample_sensor_df: pd.DataFrame) -> None:
        config = FeatureConfig(rolling_windows=(), lag_periods=())
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_sensor_df)
        assert "cycle_normalized" in result.columns
        assert result["cycle_normalized"].max() <= 1.0
        assert result["cycle_normalized"].min() >= 0.0
