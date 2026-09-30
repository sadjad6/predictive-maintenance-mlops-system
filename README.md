# 🏭 Predictive Maintenance MLOps System

**Predictive-maintenance ML prototype** using simulated turbofan sensor data. A Prefect flow generates data, creates features and labels, compares failure-classification and remaining-useful-life (RUL) models, and saves trained models. FastAPI endpoints and a Plotly Dash demonstration dashboard are included.

[![CI](https://github.com/sadjad6/predictive-maintenance-mlops-system/actions/workflows/ci.yml/badge.svg)](https://github.com/sadjad6/predictive-maintenance-mlops-system/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)

---

## 🎯 Business Problem

The repository demonstrates these predictive-maintenance workflows on **synthetic C-MAPSS-style data**:

- **Failure Prediction** — Classify machines likely to fail within a configurable time window
- **Remaining Useful Life (RUL)** — Estimate how many operating cycles remain before failure
- **Anomaly Endpoint** — Flag unusual sensor values with a heuristic z-score rule
- **What-If Simulation** — Explore maintenance scenarios with assumption-based formulas
- **Business KPIs** — Explore illustrative cost, downtime, and ROI assumptions

## 🏗️ Architecture

```text
Synthetic turbofan sensor data → validation → features and labels
                                           ↓
                                 Prefect training flow
                                           ↓
                     Classification and RUL model comparison
                                           ↓
                                  Saved model artifacts

Separate demonstrations: FastAPI endpoints · Plotly Dash UI
Deployment configuration: Docker Compose · Cloud Run workflow
```

The default training flow compares logistic regression, random forest, XGBoost, and LightGBM classifiers, plus random forest, XGBoost, and LightGBM RUL regressors. LSTM and anomaly-model classes are present, but are not part of this default flow. The Power BI and Tableau folders contain dashboard specifications, not finished workbook files.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/sadjad6/predictive-maintenance-mlops-system.git
cd predictive-maintenance-mlops-system

# Install dependencies
uv sync --all-extras

# Generate sample data and run the pipeline
uv run python -m src.pipeline.flows

# Start the API
uv run uvicorn src.api.app:app --reload --port 8000

# Start the dashboard
uv run python -m src.dashboards.web_dashboard
```

The API loader looks for `models/best_classifier` and `models/best_regressor`; the default training flow saves models under their individual names. Until those paths are supplied, failure and RUL endpoints use the API's heuristic fallback. The dashboard displays seeded demonstration data.

### Docker

```bash
# Build and run all services
docker-compose -f docker/docker-compose.yml up --build

# API available at http://localhost:8000
# Dashboard available at http://localhost:8050
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check & model status |
| `POST` | `/api/v1/predict/failure` | Predict failure probability |
| `POST` | `/api/v1/predict/rul` | Estimate remaining useful life |
| `POST` | `/api/v1/detect/anomaly` | Detect sensor anomalies |
| `POST` | `/api/v1/simulate/what-if` | Run maintenance scenarios |
| `POST` | `/api/v1/predict/batch` | Batch predictions |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict/failure \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1,
    "cycle": 150,
    "sensor_temperature": 545.0,
    "sensor_vibration": 0.035,
    "sensor_pressure": 13.5,
    "sensor_rotation_speed": 8500.0
  }'
```

## 📊 Models & Results

The training flow uses time-series cross-validation and computes classification ROC-AUC/F1 and RUL RMSE/MAE. Run it to produce results for the generated dataset. The repository does not include a reproducible benchmark report or saved model artifacts supporting fixed performance figures.

## 💰 Business Impact

The dashboard demonstrates cost and ROI calculations using configurable assumptions and seeded example predictions. Its figures are scenarios, not measured savings from an industrial deployment.

## 🔧 Technology Stack

- **Default training flow**: scikit-learn, XGBoost, LightGBM
- **API**: FastAPI, Pydantic v2, Uvicorn
- **Dashboard**: Plotly Dash
- **Orchestration**: Prefect
- **Explainability**: SHAP
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Deployment configuration**: GCP Cloud Run, Cloud Build
- **Quality**: pytest, ruff, mypy

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v --cov=src

# Run specific test suite
uv run pytest tests/test_api/ -v
uv run pytest tests/test_models/ -v

# Lint
uv run ruff check src/ tests/
```

## 📄 License

No license file is present in this repository. Add one before stating reuse terms.


