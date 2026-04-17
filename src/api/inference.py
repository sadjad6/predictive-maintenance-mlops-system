"""Inference pipeline for real-time predictions.

Loads trained models, applies feature engineering, and generates
predictions with confidence intervals and explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.config import AppConfig, get_config
from src.models.base import BaseModel, SklearnModelWrapper

if TYPE_CHECKING:
    from src.api.schemas import SensorReading

RISK_THRESHOLDS = {"critical": 0.8, "high": 0.6, "medium": 0.3}


class InferencePipeline:
    """End-to-end inference pipeline for predictions.

    Handles sensor reading → feature vector → model prediction
    with caching for model loading.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._classifier: BaseModel | None = None
        self._regressor: BaseModel | None = None
        self._feature_names: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._classifier is not None

    def load_models(self, models_dir: str | Path | None = None) -> None:
        """Load trained models from disk."""
        models_dir = Path(models_dir or self.config.model.models_dir)

        clf_path = models_dir / "best_classifier"
        reg_path = models_dir / "best_regressor"

        if clf_path.exists():
            self._classifier = self._load_sklearn_model(clf_path, "best_classifier")
            logger.info("Loaded classifier from {}", clf_path)

        if reg_path.exists():
            self._regressor = self._load_sklearn_model(reg_path, "best_regressor")
            logger.info("Loaded regressor from {}", reg_path)

        if self._classifier:
            self._feature_names = self._classifier.metadata.feature_names

    def predict_failure(self, reading: SensorReading) -> dict:
        """Predict failure probability for a sensor reading."""
        features = self._reading_to_features(reading)

        if self._classifier is None:
            return self._mock_failure_prediction(reading, features)

        proba = self._classifier.predict_proba(features)
        failure_prob = float(proba[0, 1]) if proba is not None else 0.5

        return {
            "engine_id": reading.engine_id,
            "failure_probability": round(failure_prob, 4),
            "failure_predicted": failure_prob > 0.5,
            "risk_level": _classify_risk(failure_prob),
            "confidence": round(1.0 - abs(0.5 - failure_prob) * 2, 4),
            "top_risk_factors": [],
            "recommended_action": _recommend_action(failure_prob),
        }

    def predict_rul(self, reading: SensorReading) -> dict:
        """Estimate remaining useful life."""
        features = self._reading_to_features(reading)

        if self._regressor is None:
            return self._mock_rul_prediction(reading, features)

        rul = float(self._regressor.predict(features)[0])
        margin = max(rul * 0.15, 5.0)

        urgency = "immediate" if rul < 10 else "soon" if rul < 30 else "scheduled"
        return {
            "engine_id": reading.engine_id,
            "estimated_rul_cycles": round(rul, 1),
            "rul_lower_bound": round(max(0, rul - margin), 1),
            "rul_upper_bound": round(rul + margin, 1),
            "confidence": round(max(0.5, 1.0 - margin / max(rul, 1)), 4),
            "maintenance_urgency": urgency,
        }

    def _reading_to_features(self, reading: SensorReading) -> np.ndarray:
        """Convert a SensorReading to a feature vector."""
        base_features = [
            reading.sensor_temperature,
            reading.sensor_vibration,
            reading.sensor_pressure,
            reading.sensor_rotation_speed,
            reading.sensor_voltage,
            reading.sensor_current,
            reading.op_setting_1,
            reading.op_setting_2,
            reading.op_setting_3,
        ]
        for i in range(7, 22):
            key = f"sensor_{i:02d}"
            base_features.append(reading.additional_sensors.get(key, 0.0))
        return np.array(base_features).reshape(1, -1)

    def _load_sklearn_model(self, path: Path, name: str) -> SklearnModelWrapper:
        """Load a sklearn-wrapped model from disk."""
        model = SklearnModelWrapper(name, "classification", None)
        model.load(path)
        return model

    def _mock_failure_prediction(
        self,
        reading: SensorReading,
        features: np.ndarray,
    ) -> dict:
        """Generate a heuristic prediction when no model is loaded."""
        temp_risk = max(0, (reading.sensor_temperature - 550) / 50)
        vib_risk = max(0, (reading.sensor_vibration - 0.04) / 0.03)
        prob = min(1.0, (temp_risk + vib_risk) / 2 * 0.8)

        return {
            "engine_id": reading.engine_id,
            "failure_probability": round(prob, 4),
            "failure_predicted": prob > 0.5,
            "risk_level": _classify_risk(prob),
            "confidence": 0.6,
            "top_risk_factors": [
                {"sensor_temperature": round(temp_risk, 3)},
                {"sensor_vibration": round(vib_risk, 3)},
            ],
            "recommended_action": _recommend_action(prob),
        }

    def _mock_rul_prediction(
        self,
        reading: SensorReading,
        features: np.ndarray,
    ) -> dict:
        """Generate a heuristic RUL when no model is loaded."""
        rul = max(5, 150 - reading.cycle * 0.5)
        return {
            "engine_id": reading.engine_id,
            "estimated_rul_cycles": round(rul, 1),
            "rul_lower_bound": round(max(0, rul * 0.8), 1),
            "rul_upper_bound": round(rul * 1.2, 1),
            "confidence": 0.6,
            "maintenance_urgency": "immediate" if rul < 10 else "soon" if rul < 30 else "scheduled",
        }


def _classify_risk(probability: float) -> str:
    """Map failure probability to risk level."""
    if probability >= RISK_THRESHOLDS["critical"]:
        return "critical"
    if probability >= RISK_THRESHOLDS["high"]:
        return "high"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _recommend_action(probability: float) -> str:
    """Generate maintenance recommendation based on risk."""
    if probability >= 0.8:
        return "IMMEDIATE: Schedule emergency maintenance within 24 hours"
    if probability >= 0.6:
        return "URGENT: Plan maintenance within the next 3 days"
    if probability >= 0.3:
        return "MONITOR: Increase inspection frequency"
    return "NORMAL: Continue standard maintenance schedule"
