"""Failure labeling for supervised learning targets.

Creates binary failure labels (within prediction window) and
Remaining Useful Life (RUL) targets with strict temporal ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.config import DataConfig, get_config
from src.constants import COL_CYCLE, COL_ENGINE_ID, COL_FAILURE_LABEL, COL_RUL

if TYPE_CHECKING:
    import pandas as pd


class FailureLabeler:
    """Creates ML targets from engine lifecycle data.

    Computes:
    - RUL: Remaining cycles until failure for each observation
    - Binary failure label: 1 if failure within configurable window

    All computations are per-engine with strict temporal guarantees.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or get_config().data

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL and failure labels to the DataFrame.

        Args:
            df: Sensor data with engine_id and cycle columns.

        Returns:
            DataFrame with added 'rul' and 'failure_within_window' columns.
        """
        logger.info(
            "Adding failure labels with window={} cycles",
            self.config.failure_window,
        )

        df = df.sort_values([COL_ENGINE_ID, COL_CYCLE]).reset_index(drop=True)
        df = self._compute_rul(df)
        df = self._compute_failure_label(df)

        positive_ratio = df[COL_FAILURE_LABEL].mean()
        logger.info(
            "Label distribution: {:.1%} positive (failure within window)",
            positive_ratio,
        )
        return df

    def _compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Remaining Useful Life for each observation.

        RUL = max_cycle_for_engine - current_cycle
        The last cycle for each engine is assumed to be the failure point.
        """
        max_cycles = df.groupby(COL_ENGINE_ID)[COL_CYCLE].transform("max")
        df[COL_RUL] = max_cycles - df[COL_CYCLE]
        return df

    def _compute_failure_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute binary failure label: 1 if within failure window.

        A record is labeled 1 if RUL <= failure_window, meaning
        failure is expected within the next N cycles.
        """
        df[COL_FAILURE_LABEL] = (df[COL_RUL] <= self.config.failure_window).astype(int)
        return df

    def clip_rul(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
        """Clip RUL values to a maximum (piecewise linear degradation).

        In practice, sensors don't show degradation at the very start
        of an engine's life. Clipping RUL caps the regression target.

        Args:
            df: DataFrame with RUL column.
            max_rul: Maximum RUL value to clip at.

        Returns:
            DataFrame with clipped RUL.
        """
        df[COL_RUL] = df[COL_RUL].clip(upper=max_rul)
        logger.info("Clipped RUL to max value of {}", max_rul)
        return df
