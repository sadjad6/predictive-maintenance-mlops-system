"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Input sensor readings for a single observation."""

    engine_id: int = Field(..., description="Unique engine identifier")
    cycle: int = Field(..., ge=1, description="Current operating cycle")
    sensor_temperature: float = Field(..., description="Temperature (°F)")
    sensor_vibration: float = Field(..., description="Vibration (g)")
    sensor_pressure: float = Field(..., description="Pressure (psi)")
    sensor_rotation_speed: float = Field(..., description="Rotation speed (RPM)")
    sensor_voltage: float = Field(230.0, description="Voltage (V)")
    sensor_current: float = Field(15.0, description="Current (A)")
    op_setting_1: float = Field(0.0, description="Operational setting 1")
    op_setting_2: float = Field(0.0, description="Operational setting 2")
    op_setting_3: float = Field(100.0, description="Operational setting 3")
    additional_sensors: dict[str, float] = Field(
        default_factory=dict, description="Additional sensor readings"
    )


class FailurePredictionResponse(BaseModel):
    """Response for failure prediction endpoint."""

    engine_id: int
    failure_probability: float = Field(..., ge=0, le=1)
    failure_predicted: bool
    risk_level: str = Field(..., description="low, medium, high, critical")
    confidence: float = Field(..., ge=0, le=1)
    top_risk_factors: list[dict[str, float]] = Field(default_factory=list)
    recommended_action: str = ""


class RULResponse(BaseModel):
    """Response for Remaining Useful Life estimation."""

    engine_id: int
    estimated_rul_cycles: float
    rul_lower_bound: float
    rul_upper_bound: float
    confidence: float = Field(..., ge=0, le=1)
    maintenance_urgency: str = ""


class AnomalyResponse(BaseModel):
    """Response for anomaly detection endpoint."""

    engine_id: int
    is_anomaly: bool
    anomaly_score: float
    anomalous_sensors: list[str] = Field(default_factory=list)


class WhatIfRequest(BaseModel):
    """Request for what-if simulation."""

    engine_id: int
    current_cycle: int = Field(..., ge=1)
    maintenance_at_cycle: int = Field(..., ge=1)
    sensor_readings: SensorReading


class WhatIfResponse(BaseModel):
    """Response for what-if simulation."""

    engine_id: int
    current_failure_probability: float
    post_maintenance_failure_probability: float
    probability_reduction: float
    estimated_cost_savings: float
    recommended: bool
    explanation: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    model_name: str = ""
    version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    readings: list[SensorReading]
