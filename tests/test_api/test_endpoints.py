"""Tests for API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_reading() -> dict:
    return {
        "engine_id": 1,
        "cycle": 150,
        "sensor_temperature": 545.0,
        "sensor_vibration": 0.035,
        "sensor_pressure": 13.5,
        "sensor_rotation_speed": 8500.0,
        "sensor_voltage": 225.0,
        "sensor_current": 16.0,
        "op_setting_1": 0.001,
        "op_setting_2": 0.0003,
        "op_setting_3": 100.0,
    }


class TestHealthEndpoint:
    """Tests for health check."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_format(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data


class TestFailurePredictionEndpoint:
    """Tests for failure prediction."""

    def test_predict_failure_returns_200(self, client: TestClient, sample_reading: dict) -> None:
        response = client.post("/api/v1/predict/failure", json=sample_reading)
        assert response.status_code == 200

    def test_predict_failure_response_format(self, client: TestClient, sample_reading: dict) -> None:
        data = client.post("/api/v1/predict/failure", json=sample_reading).json()
        assert "failure_probability" in data
        assert "risk_level" in data
        assert "recommended_action" in data
        assert 0 <= data["failure_probability"] <= 1
        assert data["risk_level"] in ("low", "medium", "high", "critical")

    def test_predict_failure_invalid_input(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict/failure", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_failure_missing_required_fields(self, client: TestClient) -> None:
        response = client.post("/api/v1/predict/failure", json={"engine_id": 1})
        assert response.status_code == 422


class TestRULEndpoint:
    """Tests for RUL estimation."""

    def test_predict_rul_returns_200(self, client: TestClient, sample_reading: dict) -> None:
        response = client.post("/api/v1/predict/rul", json=sample_reading)
        assert response.status_code == 200

    def test_predict_rul_response_format(self, client: TestClient, sample_reading: dict) -> None:
        data = client.post("/api/v1/predict/rul", json=sample_reading).json()
        assert "estimated_rul_cycles" in data
        assert "rul_lower_bound" in data
        assert "rul_upper_bound" in data
        assert data["rul_lower_bound"] <= data["estimated_rul_cycles"]
        assert data["estimated_rul_cycles"] <= data["rul_upper_bound"]


class TestAnomalyEndpoint:
    """Tests for anomaly detection."""

    def test_detect_anomaly_returns_200(self, client: TestClient, sample_reading: dict) -> None:
        response = client.post("/api/v1/detect/anomaly", json=sample_reading)
        assert response.status_code == 200

    def test_detect_anomaly_response_format(self, client: TestClient, sample_reading: dict) -> None:
        data = client.post("/api/v1/detect/anomaly", json=sample_reading).json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert isinstance(data["is_anomaly"], bool)


class TestWhatIfEndpoint:
    """Tests for what-if simulation."""

    def test_whatif_returns_200(self, client: TestClient, sample_reading: dict) -> None:
        request = {
            "engine_id": 1,
            "current_cycle": 150,
            "maintenance_at_cycle": 180,
            "sensor_readings": sample_reading,
        }
        response = client.post("/api/v1/simulate/what-if", json=request)
        assert response.status_code == 200

    def test_whatif_response_format(self, client: TestClient, sample_reading: dict) -> None:
        request = {
            "engine_id": 1,
            "current_cycle": 150,
            "maintenance_at_cycle": 180,
            "sensor_readings": sample_reading,
        }
        data = client.post("/api/v1/simulate/what-if", json=request).json()
        assert "current_failure_probability" in data
        assert "estimated_cost_savings" in data
        assert "recommended" in data
        assert "explanation" in data


class TestBatchEndpoint:
    """Tests for batch prediction."""

    def test_batch_predict_returns_list(self, client: TestClient, sample_reading: dict) -> None:
        request = {"readings": [sample_reading, sample_reading]}
        response = client.post("/api/v1/predict/batch", json=request)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
