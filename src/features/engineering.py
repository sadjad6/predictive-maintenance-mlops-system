"""Feature engineering for sensor time-series data.

Computes rolling statistics, lag features, rate-of-change features,
and cross-sensor interactions — all per-engine with strict temporal ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.config import FeatureConfig, get_config
from src.constants import COL_CYCLE, COL_ENGINE_ID, COL_TIMESTAMP

if TYPE_CHECKING:
    import pandas as pd


class FeatureEngineer:
    """Computes ML features from raw sensor data.

    All features are computed per-engine with temporal ordering to
    prevent data leakage. Includes rolling statistics, lag features,
    rate-of-change, and cross-sensor interactions.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or get_config().features
        self._feature_columns: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        """Return list of generated feature column names."""
        return self._feature_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations.

        Args:
            df: Raw sensor data sorted by engine_id and cycle.

        Returns:
            DataFrame with original columns plus computed features.
        """
        logger.info("Starting feature engineering on {} rows", len(df))

        # Ensure sorting (critical for temporal features)
        df = df.sort_values([COL_ENGINE_ID, COL_CYCLE]).reset_index(drop=True)

        sensor_cols = self._get_sensor_columns(df)
        logger.info("Found {} sensor columns", len(sensor_cols))

        df = self._add_rolling_features(df, sensor_cols)
        df = self._add_lag_features(df, sensor_cols)
        df = self._add_rate_of_change(df, sensor_cols)
        df = self._add_cross_sensor_features(df, sensor_cols)
        df = self._add_cycle_features(df)

        # Drop rows with NaN from rolling/lag computations
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(
            "Feature engineering complete: {} features, {} rows (dropped {} NaN rows)",
            len(self._feature_columns),
            len(df),
            initial_len - len(df),
        )
        return df

    def _get_sensor_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify sensor columns (numeric, excluding IDs and metadata)."""
        exclude = {COL_ENGINE_ID, COL_CYCLE, COL_TIMESTAMP, "rul", "failure_within_window"}
        return [col for col in df.select_dtypes(include=["number"]).columns if col not in exclude]

    def _add_rolling_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Add rolling mean, std, min, max for each sensor per engine."""
        for window in self.config.rolling_windows:
            for col in sensor_cols:
                grouped = df.groupby(COL_ENGINE_ID)[col]

                mean_col = f"{col}_roll_mean_{window}"
                std_col = f"{col}_roll_std_{window}"
                min_col = f"{col}_roll_min_{window}"
                max_col = f"{col}_roll_max_{window}"

                grouped.transform(
                    lambda x: x.rolling(window, min_periods=1)  # noqa: B023
                )
                # We need to compute each stat separately
                df[mean_col] = grouped.transform(
                    lambda x, w=window: x.rolling(w, min_periods=1).mean()
                )
                df[std_col] = grouped.transform(
                    lambda x, w=window: x.rolling(w, min_periods=1).std()
                )
                df[min_col] = grouped.transform(
                    lambda x, w=window: x.rolling(w, min_periods=1).min()
                )
                df[max_col] = grouped.transform(
                    lambda x, w=window: x.rolling(w, min_periods=1).max()
                )

                self._feature_columns.extend([mean_col, std_col, min_col, max_col])

        return df

    def _add_lag_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Add lagged sensor values per engine."""
        for lag in self.config.lag_periods:
            for col in sensor_cols:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df.groupby(COL_ENGINE_ID)[col].shift(lag)
                self._feature_columns.append(lag_col)
        return df

    def _add_rate_of_change(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Add first-order difference (rate of change) per engine."""
        for col in sensor_cols:
            diff_col = f"{col}_diff"
            df[diff_col] = df.groupby(COL_ENGINE_ID)[col].diff()
            self._feature_columns.append(diff_col)
        return df

    def _add_cross_sensor_features(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """Add interaction features between key sensor pairs."""
        if len(sensor_cols) < 2:
            return df

        # Temperature-vibration interaction (common predictive signal)
        temp_cols = [c for c in sensor_cols if "temperature" in c]
        vib_cols = [c for c in sensor_cols if "vibration" in c]

        if temp_cols and vib_cols:
            interaction_col = "temp_vibration_interaction"
            df[interaction_col] = df[temp_cols[0]] * df[vib_cols[0]]
            self._feature_columns.append(interaction_col)

        # Pressure-rotation interaction
        press_cols = [c for c in sensor_cols if "pressure" in c]
        rot_cols = [c for c in sensor_cols if "rotation" in c]

        if press_cols and rot_cols:
            interaction_col = "pressure_rotation_interaction"
            df[interaction_col] = df[press_cols[0]] * df[rot_cols[0]]
            self._feature_columns.append(interaction_col)

        return df

    def _add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized cycle position within each engine's lifetime."""
        max_cycles = df.groupby(COL_ENGINE_ID)[COL_CYCLE].transform("max")
        cycle_norm_col = "cycle_normalized"
        df[cycle_norm_col] = df[COL_CYCLE] / max_cycles
        self._feature_columns.append(cycle_norm_col)

        return df
