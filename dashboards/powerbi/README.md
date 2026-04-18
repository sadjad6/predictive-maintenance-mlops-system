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

## How to Create the Dashboard

This dashboard is built by assembling the provided data model and measures. Follow these steps to build the report in Power BI Desktop:

### 1. Build the Data Model
The structure of your tables and their relationships is defined in `data_model.json`.

1. **Import Data:** Use "Get Data" to import your datasets (e.g., Parquet files from `data/processed/`, or live connections to the REST API). 
2. **Rename Tables & Columns:** Ensure the imported table names and column names match exactly what is specified in `data_model.json` (e.g., `Predictions`, `SensorData`, `MaintenanceLog`).
3. **Set Data Types:** Verify that the data types in Power Query match the types defined in the JSON file (Int64, Double, Boolean, DateTime).
4. **Create Relationships:** Go to the Model view in Power BI. Create relationships between the tables exactly as specified in the `"relationships"` array of the JSON file (e.g., a many-to-one relationship from `Predictions[engine_id]` to `SensorData[engine_id]`).

### 2. Add DAX Measures
Once the data model is built, you need to add the calculated measures.

1. Open the `measures.dax` file located in this directory.
2. In Power BI, right-click on the appropriate table (typically the `Predictions` table for most KPIs) and select **New Measure**.
3. Copy and paste each DAX formula from the `measures.dax` file one by one into the formula bar.
4. Ensure the measures are formatted correctly (e.g., format cost measures as Currency, percentages as Percentage).

### 3. Visualize
With the model and measures in place, you can now build the visualizations described in the **Dashboard Pages** section above using standard Power BI visual types.
