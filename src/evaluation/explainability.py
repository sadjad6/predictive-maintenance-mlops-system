"""Model explainability using SHAP values.

Provides global and local feature importance analysis
for tree-based and general ML models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.models.base import BaseModel


@dataclass
class ExplainabilityReport:
    """Report containing SHAP-based explanations."""

    global_importance: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)
    shap_values: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)


class ModelExplainer:
    """Generates SHAP-based model explanations.

    Supports tree-based models (XGBoost, LightGBM, RF) with TreeExplainer
    and falls back to KernelExplainer for other model types.
    """

    def __init__(self, model: BaseModel, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self._shap_values: np.ndarray | None = None
        self._explainer = None

    def compute_shap_values(
        self,
        x: np.ndarray | pd.DataFrame,
        max_samples: int = 500,
    ) -> ExplainabilityReport:
        """Compute SHAP values for the given data.

        Args:
            x: Feature matrix to explain.
            max_samples: Max samples for background dataset.

        Returns:
            ExplainabilityReport with importance rankings.
        """
        import shap

        x_arr = np.asarray(x)
        if len(x_arr) > max_samples:
            indices = np.random.default_rng(42).choice(
                len(x_arr),
                max_samples,
                replace=False,
            )
            x_sample = x_arr[indices]
        else:
            x_sample = x_arr

        logger.info("Computing SHAP values for {} samples", len(x_sample))

        estimator = getattr(self.model, "estimator", None)
        if estimator is not None and _is_tree_model(estimator):
            self._explainer = shap.TreeExplainer(estimator)
            shap_values = self._explainer.shap_values(x_sample)
        else:
            # Fallback to KernelExplainer
            background = shap.kmeans(x_sample, min(10, len(x_sample)))
            self._explainer = shap.KernelExplainer(
                self.model.predict,
                background,
            )
            shap_values = self._explainer.shap_values(x_sample)

        # Handle multi-output SHAP (e.g., binary classification returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        self._shap_values = shap_values

        # Global importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance = dict(zip(self.feature_names, mean_abs_shap, strict=False))
        sorted_features = sorted(importance, key=importance.get, reverse=True)

        report = ExplainabilityReport(
            global_importance=importance,
            top_features=sorted_features[:20],
            shap_values=shap_values,
            feature_names=self.feature_names,
        )

        logger.info("Top 5 features: {}", sorted_features[:5])
        return report

    def explain_single(
        self,
        x_single: np.ndarray,
        top_k: int = 10,
    ) -> dict[str, float]:
        """Explain a single prediction.

        Args:
            x_single: Single sample (1D array).
            top_k: Number of top contributing features.

        Returns:
            Dict of feature name → SHAP contribution.
        """

        if self._explainer is None:
            msg = "Must call compute_shap_values first"
            raise RuntimeError(msg)

        x_2d = x_single.reshape(1, -1)
        sv = self._explainer.shap_values(x_2d)
        if isinstance(sv, list):
            sv = sv[1]

        contributions = dict(zip(self.feature_names, sv.flatten(), strict=False))
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return dict(sorted_contribs[:top_k])

    def save_report(
        self,
        report: ExplainabilityReport,
        output_dir: str | Path,
    ) -> Path:
        """Save explainability report to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        importance_df = pd.DataFrame(
            list(report.global_importance.items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=False)

        path = output_dir / "feature_importance.csv"
        importance_df.to_csv(path, index=False)
        logger.info("Saved feature importance to {}", path)
        return path


def _is_tree_model(estimator: object) -> bool:
    """Check if estimator is a tree-based model."""
    tree_types = (
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
    )
    return type(estimator).__name__ in tree_types
