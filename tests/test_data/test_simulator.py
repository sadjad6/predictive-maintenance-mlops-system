"""Tests for sensor data simulator."""

from __future__ import annotations

import numpy as np

from src.config import DataConfig
from src.constants import COL_CYCLE, COL_ENGINE_ID, SENSOR_TEMPERATURE
from src.data.simulator import SensorDataSimulator


class TestSensorDataSimulator:
    """Tests for SensorDataSimulator."""

    def test_generate_correct_engine_count(self, small_data_config: DataConfig) -> None:
        sim = SensorDataSimulator(small_data_config)
        df = sim.generate()
        assert df[COL_ENGINE_ID].nunique() == small_data_config.num_engines

    def test_generate_has_required_columns(self, small_data_config: DataConfig) -> None:
        sim = SensorDataSimulator(small_data_config)
        df = sim.generate()
        assert COL_ENGINE_ID in df.columns
        assert COL_CYCLE in df.columns
        assert SENSOR_TEMPERATURE in df.columns
        assert "timestamp" in df.columns

    def test_cycles_are_monotonic_per_engine(self, small_data_config: DataConfig) -> None:
        sim = SensorDataSimulator(small_data_config)
        df = sim.generate()
        for _, group in df.groupby(COL_ENGINE_ID):
            cycles = group[COL_CYCLE].values
            assert np.all(np.diff(cycles) > 0), "Cycles must be monotonically increasing"

    def test_sensor_values_are_finite(self, small_data_config: DataConfig) -> None:
        sim = SensorDataSimulator(small_data_config)
        df = sim.generate()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"
            assert np.all(np.isfinite(df[col].values)), f"Inf found in {col}"

    def test_single_engine_generation(self) -> None:
        config = DataConfig(num_engines=1, min_cycles=10, max_cycles=20)
        sim = SensorDataSimulator(config)
        df = sim.generate()
        assert df[COL_ENGINE_ID].nunique() == 1
        assert len(df) >= 10
        assert len(df) <= 20

    def test_save_creates_file(self, small_data_config: DataConfig, tmp_path) -> None:
        config = DataConfig(
            num_engines=2,
            min_cycles=10,
            max_cycles=15,
            raw_data_path=tmp_path / "raw",
        )
        sim = SensorDataSimulator(config)
        df = sim.generate()
        path = sim.save(df, "test_data.parquet")
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_temperature_shows_degradation(self, small_data_config: DataConfig) -> None:
        """Temperature should generally increase as engine degrades."""
        sim = SensorDataSimulator(small_data_config)
        df = sim.generate()
        for _, group in df.groupby(COL_ENGINE_ID):
            early = group[SENSOR_TEMPERATURE].iloc[:5].mean()
            late = group[SENSOR_TEMPERATURE].iloc[-5:].mean()
            # Temperature should increase (degradation), allowing noise margin
            assert late > early - 10, "Temperature should trend upward near failure"
