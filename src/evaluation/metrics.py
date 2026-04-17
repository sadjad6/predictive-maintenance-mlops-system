"""Evaluation metrics for classification and regression tasks.

Provides comprehensive metric computation including ROC-AUC,
Precision-Recall, F1, RMSE, MAE, and cost-sensitive scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import BusinessConfig, get_config
from src.constants import (
    METRIC_COST_SCORE,
    METRIC_F1,
    METRIC_MAE,
    METRIC_PR_AUC,
    METRIC_PRECISION,
    METRIC_R2,
    METRIC_RECALL,
    METRIC_RMSE,
    METRIC_ROC_AUC,
)


@dataclass
class ClassificationReport:
    """Comprehensive classification evaluation report."""

    roc_auc: float = 0.0
    pr_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    cost_score: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            METRIC_ROC_AUC: self.roc_auc,
            METRIC_PR_AUC: self.pr_auc,
            METRIC_F1: self.f1,
            METRIC_PRECISION: self.precision,
            METRIC_RECALL: self.recall,
            METRIC_COST_SCORE: self.cost_score,
        }


@dataclass
class RegressionReport:
    """Comprehensive regression evaluation report."""

    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            METRIC_RMSE: self.rmse,
            METRIC_MAE: self.mae,
            METRIC_R2: self.r2,
        }


class MetricsCalculator:
    """Computes evaluation metrics for predictive maintenance models.

    Supports both classification (failure prediction) and regression
    (RUL estimation) with cost-sensitive evaluation.
    """

    def __init__(self, business_config: BusinessConfig | None = None) -> None:
        self.biz = business_config or get_config().business

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> ClassificationReport:
        """Evaluate classification predictions.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            ClassificationReport with all metrics.
        """
        report = ClassificationReport()

        report.f1 = f1_score(y_true, y_pred, zero_division=0)
        report.precision = precision_score(y_true, y_pred, zero_division=0)
        report.recall = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report.confusion = cm
        report.true_negatives = int(cm[0, 0])
        report.false_positives = int(cm[0, 1])
        report.false_negatives = int(cm[1, 0])
        report.true_positives = int(cm[1, 1])

        if y_proba is not None and len(np.unique(y_true)) > 1:
            report.roc_auc = roc_auc_score(y_true, y_proba)
            report.pr_auc = average_precision_score(y_true, y_proba)

        report.cost_score = self._compute_cost_score(report)

        logger.info(
            "Classification — F1: {:.4f}, ROC-AUC: {:.4f}, Recall: {:.4f}, Cost: ${:,.0f}",
            report.f1,
            report.roc_auc,
            report.recall,
            report.cost_score,
        )
        return report

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RegressionReport:
        """Evaluate regression predictions (RUL estimation).

        Args:
            y_true: Ground truth RUL values.
            y_pred: Predicted RUL values.

        Returns:
            RegressionReport with all metrics.
        """
        mse = float(np.mean((y_true - y_pred) ** 2))
        report = RegressionReport(
            rmse=sqrt(mse),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
        )

        logger.info(
            "Regression — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}",
            report.rmse,
            report.mae,
            report.r2,
        )
        return report

    def _compute_cost_score(self, report: ClassificationReport) -> float:
        """Compute total expected cost from prediction errors.

        False negatives (missed failures) are much more expensive
        than false positives (unnecessary maintenance).
        """
        fn_cost = (
            report.false_negatives
            * self.biz.downtime_cost_per_hour
            * self.biz.avg_repair_hours
            * self.biz.false_negative_multiplier
        )
        fp_cost = (
            report.false_positives * self.biz.maintenance_cost * self.biz.false_positive_multiplier
        )
        return fn_cost + fp_cost
