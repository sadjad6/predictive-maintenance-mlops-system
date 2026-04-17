"""Tests for model training module."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import METRIC_F1, METRIC_ROC_AUC
from src.models.baseline import LogisticRegressionModel, RandomForestClassifierModel
from src.models.training import ModelTrainer, _compute_fold_metrics


class TestModelTraining:
    """Tests for model training pipeline."""

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((200, 10))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        return x, y

    def test_logistic_regression_trains(self, training_data) -> None:
        x, y = training_data
        model = LogisticRegressionModel()
        model.train(x, y)
        assert model.is_trained
        preds = model.predict(x)
        assert len(preds) == len(y)

    def test_random_forest_trains(self, training_data) -> None:
        x, y = training_data
        model = RandomForestClassifierModel(n_estimators=10)
        model.train(x, y)
        assert model.is_trained
        assert model.feature_importances is not None
        assert len(model.feature_importances) == x.shape[1]

    def test_predict_proba_returns_probabilities(self, training_data) -> None:
        x, y = training_data
        model = RandomForestClassifierModel(n_estimators=10)
        model.train(x, y)
        proba = model.predict_proba(x)
        assert proba is not None
        assert proba.shape == (len(x), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_model_save_load_roundtrip(self, training_data, tmp_path) -> None:
        x, y = training_data
        model = RandomForestClassifierModel(n_estimators=10)
        model.train(x, y)
        preds_before = model.predict(x)

        model.save(tmp_path / "test_model")

        model2 = RandomForestClassifierModel(n_estimators=10)
        model2.load(tmp_path / "test_model")
        preds_after = model2.predict(x)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_trainer_cross_validation(self, training_data, small_model_config) -> None:
        x, y = training_data
        trainer = ModelTrainer(small_model_config)
        model = RandomForestClassifierModel(n_estimators=10)
        result = trainer.train_and_evaluate(model, x, y)

        assert result.model_name == model.model_name
        assert METRIC_ROC_AUC in result.cv_scores
        assert METRIC_F1 in result.cv_scores
        assert len(result.cv_scores[METRIC_ROC_AUC]) == small_model_config.n_cv_splits

    def test_compute_fold_metrics_classification(self, training_data) -> None:
        x, y = training_data
        model = RandomForestClassifierModel(n_estimators=10)
        model.train(x, y)
        metrics = _compute_fold_metrics(model, x, y, is_classification=True)
        assert METRIC_ROC_AUC in metrics
        assert METRIC_F1 in metrics
        assert 0 <= metrics[METRIC_F1] <= 1

    def test_untrained_model_cannot_register(self, small_model_config) -> None:
        from src.models.base import ModelRegistry

        registry = ModelRegistry(small_model_config)
        model = LogisticRegressionModel()
        with pytest.raises(ValueError, match="untrained"):
            registry.register(model)
