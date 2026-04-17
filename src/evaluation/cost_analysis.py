"""Cost-sensitive evaluation and business impact analysis.

Translates model predictions into financial metrics: expected cost
savings, break-even analysis, and ROI of predictive maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.config import BusinessConfig, get_config


@dataclass
class CostReport:
    """Business cost analysis report."""

    total_cost_with_model: float = 0.0
    total_cost_without_model: float = 0.0
    cost_savings: float = 0.0
    savings_percentage: float = 0.0
    prevented_failures: int = 0
    unnecessary_maintenance: int = 0
    missed_failures: int = 0
    estimated_downtime_hours_saved: float = 0.0
    roi_percentage: float = 0.0


class CostAnalyzer:
    """Analyzes the financial impact of predictive maintenance models.

    Computes cost savings by comparing reactive vs predictive
    maintenance strategies with configurable cost parameters.
    """

    def __init__(self, config: BusinessConfig | None = None) -> None:
        self.config = config or get_config().business

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_deployment_cost: float = 50_000.0,
    ) -> CostReport:
        """Compute full cost analysis comparing reactive vs predictive.

        Args:
            y_true: Ground truth failure labels.
            y_pred: Model predictions.
            model_deployment_cost: One-time cost of deploying the model.

        Returns:
            CostReport with financial metrics.
        """
        report = CostReport()

        # Confusion matrix components
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        total_failures = int(np.sum(y_true == 1))

        report.prevented_failures = tp
        report.unnecessary_maintenance = fp
        report.missed_failures = fn

        # Cost WITHOUT model (reactive — all failures cause downtime)
        report.total_cost_without_model = (
            total_failures * self.config.downtime_cost_per_hour * self.config.avg_repair_hours
        )

        # Cost WITH model
        missed_failure_cost = fn * self.config.downtime_cost_per_hour * self.config.avg_repair_hours
        preventive_maintenance_cost = tp * self.config.maintenance_cost
        unnecessary_maintenance_cost = fp * self.config.maintenance_cost

        report.total_cost_with_model = (
            missed_failure_cost + preventive_maintenance_cost + unnecessary_maintenance_cost
        )

        # Savings
        report.cost_savings = report.total_cost_without_model - report.total_cost_with_model
        if report.total_cost_without_model > 0:
            report.savings_percentage = report.cost_savings / report.total_cost_without_model * 100

        # Downtime saved
        report.estimated_downtime_hours_saved = tp * self.config.avg_repair_hours

        # ROI
        if model_deployment_cost > 0:
            report.roi_percentage = (
                (report.cost_savings - model_deployment_cost) / model_deployment_cost * 100
            )

        logger.info(
            "Cost analysis: savings=${:,.0f} ({:.1f}%), ROI={:.1f}%",
            report.cost_savings,
            report.savings_percentage,
            report.roi_percentage,
        )
        return report

    def break_even_analysis(
        self,
        cost_per_failure: float | None = None,
        maintenance_cost: float | None = None,
    ) -> dict[str, float]:
        """Compute break-even thresholds for the model.

        Returns:
            Dict with minimum recall and maximum FPR for profitability.
        """
        failure_cost = cost_per_failure or (
            self.config.downtime_cost_per_hour * self.config.avg_repair_hours
        )
        maint_cost = maintenance_cost or self.config.maintenance_cost

        # Minimum recall for the model to be cost-effective
        # Savings per TP = failure_cost - maintenance_cost
        # Cost per FP = maintenance_cost
        min_recall_for_savings = maint_cost / failure_cost
        cost_ratio = failure_cost / maint_cost

        return {
            "min_recall": min_recall_for_savings,
            "cost_ratio_failure_to_maintenance": cost_ratio,
            "failure_cost": failure_cost,
            "maintenance_cost": maint_cost,
        }
