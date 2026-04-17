"""Central configuration for the Predictive Maintenance system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.constants import (
    DEFAULT_DOWNTIME_COST_PER_HOUR,
    DEFAULT_FAILURE_WINDOW_CYCLES,
    DEFAULT_FALSE_NEGATIVE_MULTIPLIER,
    DEFAULT_FALSE_POSITIVE_MULTIPLIER,
    DEFAULT_MAINTENANCE_COST,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data generation and ingestion."""

    num_engines: int = 100
    min_cycles: int = 128
    max_cycles: int = 362
    failure_window: int = DEFAULT_FAILURE_WINDOW_CYCLES
    raw_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    features_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "features")
    validation_z_threshold: float = 4.0
    max_missing_ratio: float = 0.05


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    rolling_windows: tuple[int, ...] = (5, 10, 20)
    lag_periods: tuple[int, ...] = (1, 3, 5)
    min_variance_threshold: float = 0.01
    max_correlation_threshold: float = 0.95


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model training."""

    test_split_ratio: float = 0.2
    n_cv_splits: int = 5
    random_state: int = 42
    n_optuna_trials: int = 50
    lstm_sequence_length: int = 30
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 50
    lstm_batch_size: int = 64
    lstm_learning_rate: float = 0.001
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")


@dataclass(frozen=True)
class BusinessConfig:
    """Business cost parameters for cost-sensitive evaluation."""

    downtime_cost_per_hour: float = DEFAULT_DOWNTIME_COST_PER_HOUR
    maintenance_cost: float = DEFAULT_MAINTENANCE_COST
    false_negative_multiplier: float = DEFAULT_FALSE_NEGATIVE_MULTIPLIER
    false_positive_multiplier: float = DEFAULT_FALSE_POSITIVE_MULTIPLIER
    avg_repair_hours: float = 8.0
    currency_symbol: str = "$"


@dataclass(frozen=True)
class APIConfig:
    """Configuration for the FastAPI service."""

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))  # noqa: S104
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "models/best_model.joblib")
    )


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the Dash dashboard."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8050
    debug: bool = False
    theme: str = "dark"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def get_config() -> AppConfig:
    """Factory function to create the application configuration."""
    return AppConfig()
