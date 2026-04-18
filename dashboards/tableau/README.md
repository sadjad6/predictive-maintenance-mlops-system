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

## How to Create the Dashboard

This dashboard is built by assembling the provided data model and Level of Detail (LOD) expressions. Follow these steps to build the report in Tableau Desktop:

### 1. Connect and Model the Data
The structure of your tables and their relationships is defined in `../powerbi/data_model.json`. Even though it's in the Power BI folder, the same logical model applies.

1. **Connect to Data:** Open Tableau Desktop and connect to your data sources (e.g., Parquet files in `data/processed/` or Web Data Connector for the REST API).
2. **Create the Data Source:** Drag your tables into the data model canvas.
3. **Define Relationships:** Create relationships (not necessarily physical joins, Tableau's logical relationships are preferred) between the tables based on the `engine_id` field, as specified in the JSON data model.

### 2. Add Calculated Fields (LODs)
Once the data source is ready, you need to add the necessary calculated fields, specifically the Level of Detail expressions required to handle the time-series snapshot data.

1. Open the `calculated_fields.txt` file located in this directory.
2. In Tableau, navigate to any worksheet.
3. For each formula in the text file, click the dropdown arrow next to the Search bar in the Data pane and select **Create Calculated Field...**
4. Name the field exactly as written in brackets (e.g., `Latest Cycle per Engine`) and paste the corresponding formula below it into the calculation editor.
5. Pay special attention to the LOD expressions (the ones wrapped in `{ }`), as these are critical for ensuring your KPIs only reflect the most recent state of each engine.

### 3. Build the Worksheets and Dashboard
With your data source modeled and calculated fields created, you can now build the worksheets as outlined in the **Dashboard Worksheets** section above.

1. Create individual worksheets for Sensor Trends, Failure Patterns, and KPIs using the newly created calculated fields.
2. Combine these worksheets into a single Dashboard.
3. Add dashboard actions (e.g., filter actions) to allow users to click on an engine in a summary view and see its specific sensor trends in another view.
