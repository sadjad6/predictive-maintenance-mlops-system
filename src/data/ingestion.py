"""Data ingestion pipeline for sensor data.

Handles reading raw sensor data from various sources,
schema enforcement, and storage in processed format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import DataConfig, get_config
from src.constants import COL_CYCLE, COL_ENGINE_ID, COL_TIMESTAMP

EXPECTED_MIN_COLUMNS = 10


class DataIngestor:
    """Reads, validates schema, and persists sensor data.

    Supports batch ingestion from CSV and Parquet files with
    schema enforcement and deduplication.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or get_config().data

    def ingest_from_file(self, file_path: str | Path) -> pd.DataFrame:
        """Ingest sensor data from a file (CSV or Parquet).

        Args:
            file_path: Path to the source file.

        Returns:
            Ingested and schema-enforced DataFrame.

        Raises:
            ValueError: If the file format is unsupported or schema invalid.
        """
        file_path = Path(file_path)
        logger.info("Ingesting data from {}", file_path)

        df = self._read_file(file_path)
        df = self._enforce_schema(df)
        df = self._deduplicate(df)

        logger.info(
            "Ingested {} records for {} engines",
            len(df),
            df[COL_ENGINE_ID].nunique(),
        )
        return df

    def ingest_batch(self, directory: str | Path) -> pd.DataFrame:
        """Ingest all data files from a directory.

        Args:
            directory: Path to directory containing data files.

        Returns:
            Concatenated DataFrame from all files.
        """
        directory = Path(directory)
        frames: list[pd.DataFrame] = []

        for file_path in sorted(directory.iterdir()):
            if file_path.suffix in (".csv", ".parquet"):
                frames.append(self.ingest_from_file(file_path))

        if not frames:
            msg = f"No data files found in {directory}"
            raise FileNotFoundError(msg)

        return pd.concat(frames, ignore_index=True)

    def save_processed(
        self,
        df: pd.DataFrame,
        filename: str = "processed_data.parquet",
    ) -> Path:
        """Save processed data to the processed directory."""
        output_dir = Path(self.config.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info("Saved processed data to {}", output_path)
        return output_path

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read a data file based on its extension."""
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)

        msg = f"Unsupported file format: {file_path.suffix}"
        raise ValueError(msg)

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce required columns and data types."""
        required_cols = [COL_ENGINE_ID, COL_CYCLE]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)

        if len(df.columns) < EXPECTED_MIN_COLUMNS:
            msg = f"Expected at least {EXPECTED_MIN_COLUMNS} columns, got {len(df.columns)}"
            raise ValueError(msg)

        df[COL_ENGINE_ID] = df[COL_ENGINE_ID].astype(int)
        df[COL_CYCLE] = df[COL_CYCLE].astype(int)

        if COL_TIMESTAMP in df.columns:
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        before = len(df)
        df = df.drop_duplicates(subset=[COL_ENGINE_ID, COL_CYCLE], keep="last")
        removed = before - len(df)
        if removed > 0:
            logger.warning("Removed {} duplicate records", removed)
        return df
