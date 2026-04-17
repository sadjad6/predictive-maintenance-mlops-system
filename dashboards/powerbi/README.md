# Power BI Dashboard — Connection Guide

## Overview

This directory contains the data model specification and DAX measures
for the **Executive Predictive Maintenance Dashboard** in Power BI.

## Data Sources

Connect Power BI to the following endpoints:

### REST API (Live Connection)
```
Base URL: http://<api-host>:8000/api/v1
Endpoints:
  - GET /health              → System status
  - POST /predict/failure    → Failure predictions
  - POST /predict/rul        → RUL estimates
  - POST /predict/batch      → Batch predictions
```

### Static Data (Import Mode)
- `data/processed/processed_data.parquet` — Historical sensor data
- `data/features/featured_data.parquet` — Engineered features
- `reports/model_comparison.csv` — Model performance metrics
- `reports/feature_importance.csv` — SHAP feature importance

## Dashboard Pages

### 1. Executive Summary
- Fleet health score gauge
- Machines at risk (critical/high)
- Monthly cost savings trend
- Maintenance compliance rate

### 2. Failure Risk Overview
- Risk heatmap by machine
- Failure probability distribution
- Time-to-failure trends
- Top 10 at-risk machines table

### 3. Financial Impact
- Downtime cost avoidance (cumulative)
- Maintenance cost vs. reactive cost comparison
- ROI over time
- Cost per machine category

### 4. Maintenance Planner
- Gantt chart of scheduled maintenance
- Priority matrix (urgency vs. impact)
- Resource allocation view

## DAX Measures

Key measures to create in Power BI:

```dax
// Health Score
Health Score = 
    AVERAGE('Predictions'[health_score]) * 100

// Annual Savings
Annual Savings = 
    SUMX(
        'Predictions',
        'Predictions'[failure_probability] * [Avg Failure Cost]
        - [Maintenance Cost]
    )

// Machines at Risk
Critical Machines = 
    COUNTROWS(
        FILTER('Predictions', 'Predictions'[risk_level] = "critical")
    )
```

## Setup Instructions

1. Open Power BI Desktop
2. Get Data → Web → Enter API base URL
3. Import Parquet files from `data/` directory
4. Create relationships using `engine_id` as key
5. Apply the DAX measures above
6. Use the data model from `data_model.json`
