"""API route definitions for prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import (
    AnomalyResponse,
    BatchPredictionRequest,
    FailurePredictionResponse,
    HealthResponse,
    RULResponse,
    SensorReading,
    WhatIfRequest,
    WhatIfResponse,
)

router = APIRouter()


def _get_pipeline():
    """Lazy import to avoid circular dependency."""
    from src.api.app import pipeline
    return pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    pipe = _get_pipeline()
    return HealthResponse(
        status="healthy",
        model_loaded=pipe.is_loaded,
        model_name="predictive_maintenance_v1",
        version="1.0.0",
    )


@router.post("/predict/failure", response_model=FailurePredictionResponse)
async def predict_failure(reading: SensorReading) -> FailurePredictionResponse:
    """Predict failure risk for given sensor readings."""
    pipe = _get_pipeline()
    result = pipe.predict_failure(reading)
    return FailurePredictionResponse(**result)


@router.post("/predict/rul", response_model=RULResponse)
async def predict_rul(reading: SensorReading) -> RULResponse:
    """Estimate Remaining Useful Life for an engine."""
    pipe = _get_pipeline()
    result = pipe.predict_rul(reading)
    return RULResponse(**result)


@router.post("/detect/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(reading: SensorReading) -> AnomalyResponse:
    """Detect anomalies in sensor readings."""
    pipe = _get_pipeline()
    # Heuristic anomaly detection when model not loaded
    temp_zscore = abs(reading.sensor_temperature - 520) / 30
    vib_zscore = abs(reading.sensor_vibration - 0.02) / 0.01

    is_anomaly = temp_zscore > 3 or vib_zscore > 3
    score = max(temp_zscore, vib_zscore) / 5.0

    anomalous = []
    if temp_zscore > 3:
        anomalous.append("sensor_temperature")
    if vib_zscore > 3:
        anomalous.append("sensor_vibration")

    return AnomalyResponse(
        engine_id=reading.engine_id,
        is_anomaly=is_anomaly,
        anomaly_score=round(min(score, 1.0), 4),
        anomalous_sensors=anomalous,
    )


@router.post("/simulate/what-if", response_model=WhatIfResponse)
async def what_if_simulation(request: WhatIfRequest) -> WhatIfResponse:
    """Simulate maintenance scenarios and cost impact."""
    pipe = _get_pipeline()
    current = pipe.predict_failure(request.sensor_readings)
    current_prob = current["failure_probability"]

    # Simulate post-maintenance: reduced risk proportional to how early maintenance is done
    cycles_until_maintenance = max(1, request.maintenance_at_cycle - request.current_cycle)
    reduction_factor = min(0.9, cycles_until_maintenance * 0.03)
    post_prob = current_prob * (1.0 - reduction_factor)

    cost_savings = (current_prob - post_prob) * 80_000  # Expected value of prevented failure
    recommended = cost_savings > 2_000  # More than maintenance cost

    return WhatIfResponse(
        engine_id=request.engine_id,
        current_failure_probability=round(current_prob, 4),
        post_maintenance_failure_probability=round(post_prob, 4),
        probability_reduction=round(current_prob - post_prob, 4),
        estimated_cost_savings=round(cost_savings, 2),
        recommended=recommended,
        explanation=(
            f"Performing maintenance at cycle {request.maintenance_at_cycle} "
            f"reduces failure risk by {(current_prob - post_prob) * 100:.1f}%, "
            f"with estimated savings of ${cost_savings:,.0f}."
        ),
    )


@router.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest) -> list[FailurePredictionResponse]:
    """Batch failure prediction for multiple readings."""
    pipe = _get_pipeline()
    results = []
    for reading in request.readings:
        result = pipe.predict_failure(reading)
        results.append(FailurePredictionResponse(**result))
    return results
