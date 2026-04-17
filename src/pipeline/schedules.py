"""Prefect deployment schedules for automated pipeline execution.

Defines cron-based schedules and deployment configurations
for production pipeline automation.
"""

from __future__ import annotations

from prefect.client.schemas.schedules import CronSchedule

# Retrain models weekly on Sundays at 2 AM
WEEKLY_RETRAIN_SCHEDULE = CronSchedule(cron="0 2 * * 0", timezone="UTC")

# Daily data ingestion and validation at midnight
DAILY_INGESTION_SCHEDULE = CronSchedule(cron="0 0 * * *", timezone="UTC")

# Hourly health check
HOURLY_HEALTH_SCHEDULE = CronSchedule(cron="0 * * * *", timezone="UTC")


def create_deployments() -> dict[str, dict]:
    """Create deployment configurations for Prefect.

    Returns:
        Dict mapping deployment name to configuration.
    """
    return {
        "full-pipeline-weekly": {
            "flow": "src.pipeline.flows:full_pipeline_flow",
            "schedule": WEEKLY_RETRAIN_SCHEDULE,
            "description": "Weekly end-to-end model retraining",
            "tags": ["production", "scheduled"],
            "parameters": {},
        },
        "data-ingestion-daily": {
            "flow": "src.pipeline.flows:data_ingestion_flow",
            "schedule": DAILY_INGESTION_SCHEDULE,
            "description": "Daily sensor data ingestion and validation",
            "tags": ["production", "scheduled"],
            "parameters": {},
        },
    }
