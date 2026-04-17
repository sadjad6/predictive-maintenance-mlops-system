"""Data validation module for sensor data quality checks.

Performs statistical validation, missing value checks,
anomaly detection, and schema drift detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DataConfig, get_config
from src.constants import COL_CYCLE, COL_ENGINE_ID


@dataclass
class ValidationIssue:
    """A single validation issue found in the data."""

    severity: str  # "error", "warning", "info"
    column: str
    message: str
    affected_rows: int = 0


@dataclass
class ValidationReport:
    """Result of data validation containing all issues found."""

    is_valid: bool = True
    total_rows: int = 0
    total_columns: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue and update validity status."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.is_valid = False

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"Validation {status}: {self.total_rows} rows, "
            f"{self.error_count} errors, {self.warning_count} warnings"
        )


class DataValidator:
    """Validates sensor data for quality, completeness, and consistency.

    Checks include missing values, statistical outliers, monotonicity
    of cycle counts, and value range validation.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or get_config().data

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all validation checks on the DataFrame.

        Args:
            df: Input sensor data DataFrame.

        Returns:
            ValidationReport with all findings.
        """
        report = ValidationReport(
            total_rows=len(df), total_columns=len(df.columns)
        )

        logger.info("Validating {} rows, {} columns", len(df), len(df.columns))

        self._check_empty(df, report)
        if not report.is_valid:
            return report

        self._check_missing_values(df, report)
        self._check_required_columns(df, report)
        self._check_outliers(df, report)
        self._check_cycle_monotonicity(df, report)
        self._check_value_ranges(df, report)

        logger.info(report.summary())
        return report

    def _check_empty(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check if DataFrame is empty."""
        if df.empty:
            report.add_issue(
                ValidationIssue("error", "all", "DataFrame is empty")
            )

    def _check_missing_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Check for missing values exceeding threshold."""
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df)

            if missing_ratio > self.config.max_missing_ratio:
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        column=col,
                        message=(
                            f"Missing ratio {missing_ratio:.2%} exceeds "
                            f"threshold {self.config.max_missing_ratio:.2%}"
                        ),
                        affected_rows=int(missing_count),
                    )
                )
            elif missing_count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        column=col,
                        message=f"{missing_count} missing values found",
                        affected_rows=int(missing_count),
                    )
                )

    def _check_required_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Check that required columns exist."""
        required = [COL_ENGINE_ID, COL_CYCLE]
        for col in required:
            if col not in df.columns:
                report.add_issue(
                    ValidationIssue("error", col, f"Required column '{col}' missing")
                )

    def _check_outliers(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Detect statistical outliers using z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skip_cols = {COL_ENGINE_ID, COL_CYCLE}

        for col in numeric_cols:
            if col in skip_cols:
                continue

            values = df[col].dropna()
            if len(values) < 2:
                continue

            mean = values.mean()
            std = values.std()
            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)
            outlier_count = int((z_scores > self.config.validation_z_threshold).sum())

            if outlier_count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity="warning",
                        column=col,
                        message=(
                            f"{outlier_count} outliers detected "
                            f"(z > {self.config.validation_z_threshold})"
                        ),
                        affected_rows=outlier_count,
                    )
                )

    def _check_cycle_monotonicity(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Verify cycle numbers are monotonically increasing per engine."""
        if COL_ENGINE_ID not in df.columns or COL_CYCLE not in df.columns:
            return

        for engine_id, group in df.groupby(COL_ENGINE_ID):
            cycles = group[COL_CYCLE].values
            if not np.all(np.diff(cycles) > 0):
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        column=COL_CYCLE,
                        message=f"Non-monotonic cycles for engine {engine_id}",
                        affected_rows=len(group),
                    )
                )

    def _check_value_ranges(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Check for physically impossible sensor values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skip_cols = {COL_ENGINE_ID, COL_CYCLE}

        for col in numeric_cols:
            if col in skip_cols:
                continue

            if df[col].isna().all():
                continue

            # Check for infinite values
            inf_count = int(np.isinf(df[col].dropna().values).sum())
            if inf_count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity="error",
                        column=col,
                        message=f"{inf_count} infinite values detected",
                        affected_rows=inf_count,
                    )
                )
