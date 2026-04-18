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

### 3. Visualize the Dashboards
With the model and measures in place, you can now build the visualizations described in the **Dashboard Pages** section using standard Power BI visuals. Here are the detailed steps for creating each visual:

#### Page 1: Executive Summary
*   **Fleet Health Score Gauge:**
    *   Select the **Gauge** visual.
    *   Drag the `[Avg Health Score]` measure to the "Value" field.
    *   Set the Maximum value to 100 in the formatting pane.
*   **Machines at Risk (Critical/High):**
    *   Select the **Card** visual.
    *   Drag the `[Critical Engines Count]` measure to the "Fields" area.
    *   Format the card with a red font or background to indicate urgency.
*   **Monthly Cost Savings Trend:**
    *   Select the **Line Chart** visual.
    *   Drag `Predictions[prediction_timestamp]` to the X-axis (set to Month).
    *   Drag the `[Estimated Cost Savings]` measure to the Y-axis.

#### Page 2: Failure Risk Overview
*   **Risk Heatmap by Machine:**
    *   Select the **Matrix** visual.
    *   Drag `Predictions[engine_id]` to Rows.
    *   Drag `Predictions[cycle]` or a date hierarchy to Columns.
    *   Drag `[Avg Health Score]` to Values.
    *   Apply Conditional Formatting -> Background Color based on the health score (e.g., Red for low, Green for high).
*   **Failure Probability Distribution:**
    *   Select the **Histogram** (or Clustered Column Chart).
    *   Create bins for `Predictions[failure_probability]` (e.g., 0-10%, 10-20%) and drag to the X-axis.
    *   Drag `[Total Engines]` measure to the Y-axis.
*   **Top 10 At-Risk Machines Table:**
    *   Select the **Table** visual.
    *   Add `Predictions[engine_id]`, `[Avg Estimated RUL]`, and `[Avg Health Score]`.
    *   Filter the visual using the filter pane to show Top N (10) by `[Avg Health Score]` (Bottom) or by `[failure_probability]` (Top).

#### Page 3: Financial Impact
*   **ROI Over Time:**
    *   Select the **Area Chart**.
    *   Drag `MaintenanceLog[maintenance_date]` to the X-axis.
    *   Drag the `[Model ROI]` measure to the Y-axis.

#### Page 4: Sensor Trends
*   **Sensor Telemetry:**
    *   Select a **Line Chart** visual.
    *   Drag `SensorData[timestamp]` to the X-axis.
    *   Drag `[Avg Sensor Temperature]` and `[Avg Sensor Vibration]` to the Y-axis.
    *   Add a Slicer for `SensorData[engine_id]` so users can filter by specific machines.
