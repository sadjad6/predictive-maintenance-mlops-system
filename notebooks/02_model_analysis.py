"""Model Analysis Notebook — Predictive Maintenance.

Run with: uv run python notebooks/02_model_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import get_config
from src.data.simulator import SensorDataSimulator
from src.features.engineering import FeatureEngineer
from src.features.labeling import FailureLabeler
from src.models.baseline import LogisticRegressionModel, RandomForestClassifierModel
from src.models.gradient_boosting import XGBoostClassifierModel, LightGBMClassifierModel
from src.models.training import ModelTrainer
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.cost_analysis import CostAnalyzer
from src.evaluation.comparison import ModelComparator
from src.constants import COL_FAILURE_LABEL, METRIC_ROC_AUC


def main() -> None:
    """Run model training and analysis pipeline."""
    config = get_config()

    # ── Step 1: Generate & Prepare Data ───────────────────────────────
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)

    simulator = SensorDataSimulator(config.data)
    df = simulator.generate()

    labeler = FailureLabeler(config.data)
    df = labeler.add_labels(df)
    df = labeler.clip_rul(df)

    engineer = FeatureEngineer(config.features)
    df = engineer.transform(df)
    feature_cols = engineer.feature_columns

    print(f"Features: {len(feature_cols)}, Rows: {len(df)}")
    print(f"Positive ratio: {df[COL_FAILURE_LABEL].mean():.1%}")

    # ── Step 2: Train Models ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)

    x = df[feature_cols].values
    y = df[COL_FAILURE_LABEL].values
    trainer = ModelTrainer(config.model)

    models = [
        LogisticRegressionModel(),
        RandomForestClassifierModel(n_estimators=50),
        XGBoostClassifierModel(n_estimators=50),
        LightGBMClassifierModel(n_estimators=50),
    ]

    for model in models:
        print(f"\nTraining {model.model_name}...")
        result = trainer.train_and_evaluate(model, x, y, feature_cols)
        for metric, scores in result.cv_scores.items():
            print(f"  {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ── Step 3: Compare Models ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Model Comparison")
    print("=" * 60)

    comparator = ModelComparator(trainer.registry)
    print(comparator.summary_report(METRIC_ROC_AUC))

    # ── Step 4: Cost Analysis ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Cost Analysis")
    print("=" * 60)

    best_model = trainer.registry.get_best_model(METRIC_ROC_AUC)
    y_pred = best_model.predict(x)

    analyzer = CostAnalyzer(config.business)
    cost_report = analyzer.analyze(y, y_pred)

    print(f"\nBest model: {best_model.model_name}")
    print(f"Prevented failures: {cost_report.prevented_failures}")
    print(f"Missed failures: {cost_report.missed_failures}")
    print(f"Cost WITHOUT model: ${cost_report.total_cost_without_model:,.0f}")
    print(f"Cost WITH model: ${cost_report.total_cost_with_model:,.0f}")
    print(f"SAVINGS: ${cost_report.cost_savings:,.0f} ({cost_report.savings_percentage:.1f}%)")
    print(f"ROI: {cost_report.roi_percentage:.1f}%")

    # ── Step 5: Evaluation Metrics ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Detailed Metrics")
    print("=" * 60)

    calc = MetricsCalculator(config.business)
    proba = best_model.predict_proba(x)
    y_proba = proba[:, 1] if proba is not None else None
    clf_report = calc.evaluate_classification(y, y_pred, y_proba)

    print(f"\nROC-AUC: {clf_report.roc_auc:.4f}")
    print(f"PR-AUC: {clf_report.pr_auc:.4f}")
    print(f"F1: {clf_report.f1:.4f}")
    print(f"Precision: {clf_report.precision:.4f}")
    print(f"Recall: {clf_report.recall:.4f}")
    print(f"Total cost of errors: ${clf_report.cost_score:,.0f}")

    print("\nConfusion Matrix:")
    print(f"  TN={clf_report.true_negatives}  FP={clf_report.false_positives}")
    print(f"  FN={clf_report.false_negatives}  TP={clf_report.true_positives}")

    print("\n✅ Model analysis complete!")


if __name__ == "__main__":
    main()
