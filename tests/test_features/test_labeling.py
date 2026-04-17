"""Tests for failure labeling module."""

from __future__ import annotations

import pandas as pd

from src.config import DataConfig
from src.constants import COL_CYCLE, COL_ENGINE_ID, COL_FAILURE_LABEL, COL_RUL
from src.features.labeling import FailureLabeler


class TestFailureLabeler:
    """Tests for FailureLabeler."""

    def test_rul_computation(self, sample_sensor_df: pd.DataFrame) -> None:
        labeler = FailureLabeler()
        result = labeler.add_labels(sample_sensor_df)
        assert COL_RUL in result.columns
        # Last cycle of each engine should have RUL=0
        for _, group in result.groupby(COL_ENGINE_ID):
            last_row = group.sort_values(COL_CYCLE).iloc[-1]
            assert last_row[COL_RUL] == 0

    def test_failure_label_binary(self, sample_sensor_df: pd.DataFrame) -> None:
        labeler = FailureLabeler()
        result = labeler.add_labels(sample_sensor_df)
        assert COL_FAILURE_LABEL in result.columns
        unique_labels = set(result[COL_FAILURE_LABEL].unique())
        assert unique_labels.issubset({0, 1})

    def test_failure_window_respected(self, sample_sensor_df: pd.DataFrame) -> None:
        window = 10
        config = DataConfig(failure_window=window)
        labeler = FailureLabeler(config)
        result = labeler.add_labels(sample_sensor_df)

        # All rows with RUL <= window should be labeled 1
        positive_rows = result[result[COL_FAILURE_LABEL] == 1]
        assert all(positive_rows[COL_RUL] <= window)

        # All rows with RUL > window should be labeled 0
        negative_rows = result[result[COL_FAILURE_LABEL] == 0]
        assert all(negative_rows[COL_RUL] > window)

    def test_clip_rul_caps_values(self, sample_sensor_df: pd.DataFrame) -> None:
        labeler = FailureLabeler()
        result = labeler.add_labels(sample_sensor_df)
        result = labeler.clip_rul(result, max_rul=20)
        assert result[COL_RUL].max() <= 20

    def test_rul_non_negative(self, sample_sensor_df: pd.DataFrame) -> None:
        labeler = FailureLabeler()
        result = labeler.add_labels(sample_sensor_df)
        assert (result[COL_RUL] >= 0).all()

    def test_label_distribution_reasonable(self, sample_sensor_df: pd.DataFrame) -> None:
        """With a window of 15 and ~50 cycles, expect ~30% positive."""
        config = DataConfig(failure_window=15)
        labeler = FailureLabeler(config)
        result = labeler.add_labels(sample_sensor_df)
        positive_ratio = result[COL_FAILURE_LABEL].mean()
        assert 0.1 < positive_ratio < 0.6, f"Unexpected ratio: {positive_ratio}"

    def test_single_engine_labeling(self) -> None:
        df = pd.DataFrame({
            COL_ENGINE_ID: [1] * 10,
            COL_CYCLE: list(range(1, 11)),
        })
        config = DataConfig(failure_window=3)
        labeler = FailureLabeler(config)
        result = labeler.add_labels(df)
        # Last 4 cycles (RUL 0,1,2,3) should be positive
        assert result[COL_FAILURE_LABEL].sum() == 4
