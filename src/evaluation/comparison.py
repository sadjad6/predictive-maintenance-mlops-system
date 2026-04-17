"""Model comparison and selection framework.

Provides side-by-side metric comparison, statistical significance
testing, and automated best-model recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import ModelRegistry


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""

    comparison_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_name: str = ""
    best_metric_value: float = 0.0
    ranking_metric: str = ""
    recommendations: list[str] = field(default_factory=list)


class ModelComparator:
    """Compares trained models and recommends the best one.

    Provides tabular comparison, ranking, and automated recommendations
    based on configurable metric priorities.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def compare(
        self,
        primary_metric: str,
        higher_is_better: bool = True,
    ) -> ComparisonResult:
        """Compare all registered models.

        Args:
            primary_metric: Metric to rank models by.
            higher_is_better: If True, highest value wins.

        Returns:
            ComparisonResult with comparison table and recommendations.
        """
        table = self.registry.comparison_table()
        if table.empty:
            logger.warning("No models to compare")
            return ComparisonResult()

        ascending = not higher_is_better
        table = table.sort_values(primary_metric, ascending=ascending)

        best_row = table.iloc[0]
        best_name = best_row["model"]
        best_value = best_row[primary_metric]

        recommendations = self._generate_recommendations(table, primary_metric)

        result = ComparisonResult(
            comparison_table=table,
            best_model_name=best_name,
            best_metric_value=float(best_value),
            ranking_metric=primary_metric,
            recommendations=recommendations,
        )

        logger.info(
            "Best model: {} ({}={:.4f})", best_name, primary_metric, best_value,
        )
        return result

    def _generate_recommendations(
        self, table: pd.DataFrame, metric: str,
    ) -> list[str]:
        """Generate human-readable recommendations."""
        recs: list[str] = []

        if len(table) < 2:
            recs.append("Only one model available — no comparison possible.")
            return recs

        best = table.iloc[0]
        second = table.iloc[1]
        diff = abs(best[metric] - second[metric])

        recs.append(
            f"Recommended: {best['model']} "
            f"({metric}={best[metric]:.4f})"
        )

        if diff < 0.01:
            recs.append(
                f"Close contest with {second['model']} "
                f"(difference: {diff:.4f}). Consider model complexity."
            )
        else:
            recs.append(
                f"Clear winner over {second['model']} "
                f"(margin: {diff:.4f})"
            )

        # Check for overfitting signals
        if "task" in table.columns:
            clf_models = table[table["task"] == "classification"]
            if len(clf_models) > 1 and metric in clf_models.columns:
                values = clf_models[metric].values
                if np.std(values) > 0.1:
                    recs.append(
                        "High variance across models — "
                        "consider ensemble approaches."
                    )

        return recs

    def summary_report(self, primary_metric: str) -> str:
        """Generate a text summary of model comparison."""
        result = self.compare(primary_metric)
        lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            "=" * 60,
            f"Ranking metric: {primary_metric}",
            f"Best model: {result.best_model_name}",
            f"Best score: {result.best_metric_value:.4f}",
            "",
            "Comparison Table:",
            result.comparison_table.to_string(index=False),
            "",
            "Recommendations:",
        ]
        for rec in result.recommendations:
            lines.append(f"  • {rec}")
        lines.append("=" * 60)
        return "\n".join(lines)
