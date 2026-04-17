"""Tests for evaluation metrics module."""

from __future__ import annotations

import numpy as np

from src.evaluation.cost_analysis import CostAnalyzer
from src.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_classification_metrics_computed(self, sample_predictions) -> None:
        y_true, y_pred, y_proba = sample_predictions
        calc = MetricsCalculator()
        report = calc.evaluate_classification(y_true, y_pred, y_proba)

        assert 0 <= report.f1 <= 1
        assert 0 <= report.precision <= 1
        assert 0 <= report.recall <= 1
        assert report.roc_auc >= 0
        assert report.confusion.shape == (2, 2)

    def test_confusion_matrix_sums_correctly(self, sample_predictions) -> None:
        y_true, y_pred, y_proba = sample_predictions
        calc = MetricsCalculator()
        report = calc.evaluate_classification(y_true, y_pred, y_proba)

        total = (
            report.true_positives
            + report.false_positives
            + report.true_negatives
            + report.false_negatives
        )
        assert total == len(y_true)

    def test_cost_score_penalizes_false_negatives(self) -> None:
        """False negatives should be much more expensive than false positives."""
        calc = MetricsCalculator()

        # Scenario 1: High false negatives
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        y_pred_high_fn = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # All missed
        r1 = calc.evaluate_classification(y_true, y_pred_high_fn)

        # Scenario 2: High false positives
        y_pred_high_fp = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # All positive
        r2 = calc.evaluate_classification(y_true, y_pred_high_fp)

        # Missing failures should cost more
        assert r1.cost_score > r2.cost_score

    def test_regression_metrics_computed(self) -> None:
        y_true = np.array([100, 80, 60, 40, 20])
        y_pred = np.array([95, 75, 65, 38, 22])
        calc = MetricsCalculator()
        report = calc.evaluate_regression(y_true, y_pred)

        assert report.rmse > 0
        assert report.mae > 0
        assert report.r2 > 0  # Should be positive for decent predictions

    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        calc = MetricsCalculator()
        report = calc.evaluate_classification(y, y, None)
        assert report.f1 == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0

    def test_to_dict_contains_all_keys(self, sample_predictions) -> None:
        y_true, y_pred, y_proba = sample_predictions
        calc = MetricsCalculator()
        report = calc.evaluate_classification(y_true, y_pred, y_proba)
        d = report.to_dict()
        assert "roc_auc" in d
        assert "f1_score" in d
        assert "cost_score" in d


class TestCostAnalyzer:
    """Tests for CostAnalyzer."""

    def test_savings_positive_for_good_model(self) -> None:
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 3 TP, 1 FN
        analyzer = CostAnalyzer()
        report = analyzer.analyze(y_true, y_pred)

        assert report.cost_savings > 0
        assert report.prevented_failures == 3
        assert report.missed_failures == 1

    def test_no_savings_if_all_missed(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])  # All missed
        analyzer = CostAnalyzer()
        report = analyzer.analyze(y_true, y_pred)

        assert report.prevented_failures == 0
        assert report.missed_failures == 3
        # Total cost with model == total cost without model
        assert report.cost_savings == 0

    def test_break_even_analysis(self) -> None:
        analyzer = CostAnalyzer()
        result = analyzer.break_even_analysis()
        assert "min_recall" in result
        assert "cost_ratio_failure_to_maintenance" in result
        assert result["cost_ratio_failure_to_maintenance"] > 1
