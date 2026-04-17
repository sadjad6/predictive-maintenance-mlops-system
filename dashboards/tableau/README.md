# Tableau Dashboard — Connection Guide

## Overview

This directory contains the workbook specification for the
**Analytical Predictive Maintenance Dashboard** in Tableau.

## Data Sources

### Primary: REST API (Web Data Connector)
```
Base URL: http://<api-host>:8000/api/v1
Use Tableau Web Data Connector to pull live prediction data.
```

### Secondary: File-Based (Parquet/CSV)
- `data/processed/processed_data.parquet` — Historical sensor readings
- `data/features/featured_data.parquet` — Engineered features
- `reports/feature_importance.csv` — SHAP values

## Dashboard Worksheets

### 1. Sensor Trends Over Time
- **Type**: Line chart with dual axis
- **Dimensions**: Timestamp, Engine ID
- **Measures**: Temperature, Vibration, Pressure, Rotation Speed
- **Filters**: Engine ID selector, Date range
- **Interaction**: Hover tooltip with exact values

### 2. Failure Pattern Exploration
- **Type**: Heatmap + scatter plot
- **Dimensions**: Cycle, Sensor name
- **Measures**: Sensor values (normalized)
- **Color**: Failure proximity (RUL)
- **Insight**: Visual correlation between sensor degradation and failure

### 3. Drill-Down by Machine / Plant
- **Type**: Tree map + bar chart
- **Hierarchy**: Plant → Line → Machine → Engine
- **Measures**: Health score, Failure probability, RUL
- **Action**: Click to drill down into individual engine view

### 4. Feature Importance Visualization
- **Type**: Horizontal bar chart + Waterfall
- **Data Source**: `reports/feature_importance.csv`
- **Measures**: SHAP importance values
- **Interaction**: Click feature to see SHAP dependency plot

### 5. RUL Distribution
- **Type**: Box plot + histogram
- **Dimensions**: Risk category
- **Measures**: Estimated RUL
- **Reference lines**: Maintenance thresholds (30, 60, 90 cycles)

### 6. Anomaly Detection View
- **Type**: Scatter plot with anomaly highlighting
- **Dimensions**: Cycle
- **Measures**: Anomaly score, sensor values
- **Color**: Normal (blue) vs Anomaly (red)

## Setup Instructions

1. Open Tableau Desktop
2. Connect to Parquet files in `data/processed/`
3. Add Web Data Connector for live API data
4. Create relationships: join on `engine_id`
5. Build worksheets following specs above
6. Combine into Dashboard with filter actions
7. Publish to Tableau Server/Cloud

## Calculated Fields

```tableau
// Health Score
IF [failure_probability] < 0.3 THEN "Healthy"
ELSEIF [failure_probability] < 0.6 THEN "Warning"
ELSEIF [failure_probability] < 0.8 THEN "At Risk"
ELSE "Critical"
END

// Normalized Sensor Value
([Sensor Value] - {FIXED [Sensor Name] : MIN([Sensor Value])})
/ ({FIXED [Sensor Name] : MAX([Sensor Value])} - {FIXED [Sensor Name] : MIN([Sensor Value])})

// Days Until Maintenance
[Estimated RUL] / 24
```
