"""Data Exploration Notebook — Predictive Maintenance.

Run with: uv run python notebooks/01_data_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.simulator import SensorDataSimulator
from src.data.validation import DataValidator
from src.features.labeling import FailureLabeler
from src.config import get_config


def main() -> None:
    """Run data exploration pipeline."""
    config = get_config()

    # ── Step 1: Generate Data ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Generating Sensor Data")
    print("=" * 60)

    simulator = SensorDataSimulator(config.data)
    df = simulator.generate()

    print(f"\nDataset shape: {df.shape}")
    print(f"Engines: {df['engine_id'].nunique()}")
    print(f"Cycles range: {df['cycle'].min()} - {df['cycle'].max()}")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
    print(f"\nSample rows:\n{df.head()}")

    # ── Step 2: Validate Data ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Data Validation")
    print("=" * 60)

    validator = DataValidator(config.data)
    report = validator.validate(df)
    print(f"\n{report.summary()}")
    if report.issues:
        print(f"Issues found: {len(report.issues)}")
        for issue in report.issues[:5]:
            print(f"  [{issue.severity}] {issue.column}: {issue.message}")

    # ── Step 3: Add Labels ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Failure Labeling")
    print("=" * 60)

    labeler = FailureLabeler(config.data)
    df = labeler.add_labels(df)
    df = labeler.clip_rul(df)

    print(f"\nRUL statistics:\n{df['rul'].describe()}")
    print(f"\nFailure label distribution:\n{df['failure_within_window'].value_counts()}")
    positive_ratio = df["failure_within_window"].mean()
    print(f"Positive class ratio: {positive_ratio:.1%}")

    # ── Step 4: Statistical Summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Sensor Statistics")
    print("=" * 60)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    print(f"\n{df[sensor_cols].describe().round(2)}")

    # ── Step 5: Correlation Analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Top Correlations with RUL")
    print("=" * 60)

    correlations = df[sensor_cols + ["rul"]].corr()["rul"].drop("rul").abs()
    top_corr = correlations.sort_values(ascending=False).head(10)
    print(f"\n{top_corr}")

    # ── Save ──────────────────────────────────────────────────────────
    output_path = simulator.save(df, "explored_data.parquet")
    print(f"\nData saved to: {output_path}")
    print("\n✅ Data exploration complete!")


if __name__ == "__main__":
    main()
