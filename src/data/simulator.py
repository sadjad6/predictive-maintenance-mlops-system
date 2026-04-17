"""Realistic industrial IoT sensor data simulator.

Generates turbofan engine degradation data inspired by NASA C-MAPSS,
with configurable failure modes, regime changes, and sensor noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DataConfig, get_config
from src.constants import (
    COL_CYCLE,
    COL_ENGINE_ID,
    COL_TIMESTAMP,
    OPERATIONAL_SETTING_1,
    OPERATIONAL_SETTING_2,
    OPERATIONAL_SETTING_3,
    SENSOR_CURRENT,
    SENSOR_PRESSURE,
    SENSOR_ROTATION_SPEED,
    SENSOR_TEMPERATURE,
    SENSOR_VIBRATION,
    SENSOR_VOLTAGE,
)


@dataclass
class SensorProfile:
    """Defines baseline and degradation behavior for a sensor."""

    name: str
    base_value: float
    noise_std: float
    degradation_rate: float
    failure_spike: float


# Realistic sensor profiles based on industrial turbofan engines
DEFAULT_SENSOR_PROFILES: list[SensorProfile] = [
    SensorProfile(SENSOR_TEMPERATURE, 520.0, 3.0, 0.08, 30.0),
    SensorProfile(SENSOR_VIBRATION, 0.02, 0.005, 0.0004, 0.05),
    SensorProfile(SENSOR_PRESSURE, 14.7, 0.3, -0.015, -2.0),
    SensorProfile(SENSOR_ROTATION_SPEED, 9000.0, 50.0, -3.0, -500.0),
    SensorProfile(SENSOR_VOLTAGE, 230.0, 2.0, -0.05, -15.0),
    SensorProfile(SENSOR_CURRENT, 15.0, 0.5, 0.03, 5.0),
    SensorProfile("sensor_07", 550.0, 4.0, 0.06, 25.0),
    SensorProfile("sensor_08", 2388.0, 10.0, -1.5, -100.0),
    SensorProfile("sensor_09", 9050.0, 30.0, -2.0, -300.0),
    SensorProfile("sensor_10", 1.3, 0.02, 0.001, 0.1),
    SensorProfile("sensor_11", 47.5, 0.8, 0.04, 5.0),
    SensorProfile("sensor_12", 521.0, 3.5, 0.07, 28.0),
    SensorProfile("sensor_13", 2388.0, 8.0, -1.2, -80.0),
    SensorProfile("sensor_14", 8140.0, 40.0, -2.5, -350.0),
    SensorProfile("sensor_15", 8.44, 0.1, 0.005, 0.5),
    SensorProfile("sensor_16", 0.03, 0.002, 0.0002, 0.02),
    SensorProfile("sensor_17", 392.0, 3.0, 0.05, 20.0),
    SensorProfile("sensor_18", 2388.0, 7.0, -1.0, -70.0),
    SensorProfile("sensor_19", 100.0, 1.5, 0.03, 10.0),
    SensorProfile("sensor_20", 39.0, 0.6, 0.02, 4.0),
    SensorProfile("sensor_21", 23.4, 0.3, 0.01, 2.0),
]


class SensorDataSimulator:
    """Generates realistic turbofan engine degradation sensor data.

    Produces data mimicking NASA C-MAPSS style with multiple engines,
    each running until failure with gradually degrading sensor readings.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or get_config().data
        self._rng = np.random.default_rng(42)

    def generate(self) -> pd.DataFrame:
        """Generate sensor data for all engines.

        Returns:
            DataFrame with columns: engine_id, cycle, timestamp,
            operational settings, and 21 sensor readings.
        """
        logger.info(
            "Generating sensor data for {} engines",
            self.config.num_engines,
        )
        engine_frames: list[pd.DataFrame] = []

        for engine_id in range(1, self.config.num_engines + 1):
            max_life = self._rng.integers(self.config.min_cycles, self.config.max_cycles + 1)
            engine_df = self._generate_single_engine(engine_id, int(max_life))
            engine_frames.append(engine_df)

        combined = pd.concat(engine_frames, ignore_index=True)
        logger.info(
            "Generated {} total records across {} engines",
            len(combined),
            self.config.num_engines,
        )
        return combined

    def _generate_single_engine(self, engine_id: int, max_cycles: int) -> pd.DataFrame:
        """Generate degradation data for a single engine."""
        cycles = np.arange(1, max_cycles + 1)
        n = len(cycles)

        # Health index: 1.0 (healthy) → 0.0 (failure)
        health = 1.0 - (cycles / max_cycles) ** 1.5

        records: dict[str, np.ndarray] = {
            COL_ENGINE_ID: np.full(n, engine_id),
            COL_CYCLE: cycles,
        }

        # Operational settings (regime changes)
        records[OPERATIONAL_SETTING_1] = self._generate_regime(n, 3, 0.0, 0.01)
        records[OPERATIONAL_SETTING_2] = self._generate_regime(n, 6, 0.0, 0.001)
        records[OPERATIONAL_SETTING_3] = self._generate_regime(n, 1, 100.0, 0.5)

        # Sensor readings with degradation
        for profile in DEFAULT_SENSOR_PROFILES:
            records[profile.name] = self._generate_sensor_signal(n, health, profile)

        df = pd.DataFrame(records)

        # Add timestamps (one cycle ≈ 1 hour of operation)
        base_time = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=engine_id * 500)
        df[COL_TIMESTAMP] = pd.date_range(start=base_time, periods=n, freq="h")

        return df

    def _generate_sensor_signal(
        self,
        n: int,
        health: np.ndarray,
        profile: SensorProfile,
    ) -> np.ndarray:
        """Generate a sensor signal with degradation and noise."""
        degradation = (1.0 - health) * profile.failure_spike
        noise = self._rng.normal(0, profile.noise_std, n)

        # Add slight regime-dependent variation
        regime_noise = self._rng.normal(0, profile.noise_std * 0.3, n)

        signal = profile.base_value + degradation + noise + regime_noise

        # Add sudden spikes near failure (last 5% of life)
        spike_zone = int(n * 0.95)
        if spike_zone < n:
            spike_magnitude = abs(profile.failure_spike) * 0.3
            spike_noise = self._rng.exponential(spike_magnitude, n - spike_zone)
            if profile.degradation_rate < 0:
                signal[spike_zone:] -= spike_noise
            else:
                signal[spike_zone:] += spike_noise

        return signal

    def _generate_regime(
        self,
        n: int,
        n_regimes: int,
        center: float,
        spread: float,
    ) -> np.ndarray:
        """Generate operational regime setting with discrete levels."""
        regime_values = self._rng.uniform(
            center - spread * n_regimes,
            center + spread * n_regimes,
            n_regimes,
        )
        regime_indices = self._rng.integers(0, n_regimes, n)
        return regime_values[regime_indices]

    def save(self, df: pd.DataFrame, filename: str = "sensor_data.parquet") -> Path:
        """Save generated data to Parquet format."""
        output_dir = Path(self.config.raw_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info("Saved sensor data to {}", output_path)
        return output_path
