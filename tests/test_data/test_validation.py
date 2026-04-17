"""Tests for data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DataConfig
from src.constants import COL_CYCLE, COL_ENGINE_ID, SENSOR_TEMPERATURE
from src.data.validation import DataValidator


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_data_passes(self, sample_sensor_df: pd.DataFrame) -> None:
        validator = DataValidator()
        report = validator.validate(sample_sensor_df)
        assert report.is_valid
        assert report.error_count == 0

    def test_empty_dataframe_fails(self) -> None:
        validator = DataValidator()
        report = validator.validate(pd.DataFrame())
        assert not report.is_valid
        assert report.error_count > 0

    def test_missing_required_column_fails(self, sample_sensor_df: pd.DataFrame) -> None:
        df = sample_sensor_df.drop(columns=[COL_ENGINE_ID])
        validator = DataValidator()
        report = validator.validate(df)
        assert not report.is_valid

    def test_high_missing_ratio_fails(self, sample_sensor_df: pd.DataFrame) -> None:
        df = sample_sensor_df.copy()
        # Set 50% of temperature values to NaN
        mask = np.random.default_rng(42).random(len(df)) < 0.5
        df.loc[mask, SENSOR_TEMPERATURE] = np.nan
        config = DataConfig(max_missing_ratio=0.05)
        validator = DataValidator(config)
        report = validator.validate(df)
        assert not report.is_valid

    def test_low_missing_ratio_warns(self, sample_sensor_df: pd.DataFrame) -> None:
        df = sample_sensor_df.copy()
        # Set 1% of values to NaN
        mask = np.random.default_rng(42).random(len(df)) < 0.01
        df.loc[mask, SENSOR_TEMPERATURE] = np.nan
        validator = DataValidator()
        report = validator.validate(df)
        assert report.is_valid  # Should pass but with warnings
        assert report.warning_count > 0

    def test_infinite_values_detected(self, sample_sensor_df: pd.DataFrame) -> None:
        df = sample_sensor_df.copy()
        df.loc[0, SENSOR_TEMPERATURE] = np.inf
        validator = DataValidator()
        report = validator.validate(df)
        assert not report.is_valid

    def test_non_monotonic_cycles_detected(self) -> None:
        df = pd.DataFrame(
            {
                COL_ENGINE_ID: [1, 1, 1, 1],
                COL_CYCLE: [1, 3, 2, 4],  # Non-monotonic
                SENSOR_TEMPERATURE: [520, 521, 522, 523],
            }
        )
        validator = DataValidator()
        report = validator.validate(df)
        assert not report.is_valid

    def test_report_summary_contains_status(self, sample_sensor_df: pd.DataFrame) -> None:
        validator = DataValidator()
        report = validator.validate(sample_sensor_df)
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary
