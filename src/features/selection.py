"""Feature selection for reducing dimensionality and noise.

Provides correlation-based filtering, variance thresholding,
and mutual-information-based feature ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.config import FeatureConfig, get_config
from src.constants import TASK_CLASSIFICATION


class FeatureSelector:
    """Selects the most informative features for modeling.

    Methods:
    - Variance threshold: removes near-constant features
    - Correlation filter: removes highly correlated redundant features
    - Mutual information: ranks features by predictive power
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or get_config().features
        self._selected_features: list[str] = []
        self._feature_importances: dict[str, float] = {}

    @property
    def selected_features(self) -> list[str]:
        return self._selected_features

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(self._feature_importances)

    def fit_select(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        task: str = TASK_CLASSIFICATION,
    ) -> list[str]:
        """Run all feature selection steps and return selected features.

        Args:
            df: DataFrame with feature and target columns.
            feature_cols: List of candidate feature column names.
            target_col: Name of the target column.
            task: "classification" or "regression".

        Returns:
            List of selected feature column names.
        """
        logger.info("Starting feature selection from {} candidates", len(feature_cols))

        # Step 1: Variance threshold
        surviving = self._variance_filter(df, feature_cols)
        logger.info("After variance filter: {} features", len(surviving))

        # Step 2: Correlation filter
        surviving = self._correlation_filter(df, surviving)
        logger.info("After correlation filter: {} features", len(surviving))

        # Step 3: Mutual information ranking
        surviving = self._mutual_info_select(
            df, surviving, target_col, task, top_k=min(50, len(surviving))
        )
        logger.info("After MI selection: {} features", len(surviving))

        self._selected_features = surviving
        return surviving

    def _variance_filter(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> list[str]:
        """Remove features with variance below threshold."""
        variances = df[feature_cols].var()
        mask = variances > self.config.min_variance_threshold
        return list(variances[mask].index)

    def _correlation_filter(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> list[str]:
        """Remove one of each pair of highly correlated features."""
        if len(feature_cols) < 2:
            return feature_cols

        corr_matrix = df[feature_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        to_drop: set[str] = set()
        for col in upper_tri.columns:
            high_corr = upper_tri.index[
                upper_tri[col] > self.config.max_correlation_threshold
            ]
            to_drop.update(high_corr)

        if to_drop:
            logger.info("Dropping {} highly correlated features", len(to_drop))

        return [c for c in feature_cols if c not in to_drop]

    def _mutual_info_select(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        task: str,
        top_k: int = 50,
    ) -> list[str]:
        """Rank features by mutual information and select top-k."""
        x = df[feature_cols].fillna(0).values
        y = df[target_col].values

        mi_func = (
            mutual_info_classif
            if task == TASK_CLASSIFICATION
            else mutual_info_regression
        )
        mi_scores = mi_func(x, y, random_state=42)

        score_map = dict(zip(feature_cols, mi_scores, strict=False))
        self._feature_importances = score_map

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in ranked[:top_k]]

        return selected
