"""Shared constants for the Predictive Maintenance system."""

# ── Column Names ──────────────────────────────────────────────────────────
COL_ENGINE_ID = "engine_id"
COL_CYCLE = "cycle"
COL_RUL = "rul"
COL_FAILURE_LABEL = "failure_within_window"
COL_TIMESTAMP = "timestamp"

# Sensor column names
SENSOR_TEMPERATURE = "sensor_temperature"
SENSOR_VIBRATION = "sensor_vibration"
SENSOR_PRESSURE = "sensor_pressure"
SENSOR_ROTATION_SPEED = "sensor_rotation_speed"
SENSOR_VOLTAGE = "sensor_voltage"
SENSOR_CURRENT = "sensor_current"

OPERATIONAL_SETTING_1 = "op_setting_1"
OPERATIONAL_SETTING_2 = "op_setting_2"
OPERATIONAL_SETTING_3 = "op_setting_3"

CORE_SENSOR_COLUMNS = [
    SENSOR_TEMPERATURE,
    SENSOR_VIBRATION,
    SENSOR_PRESSURE,
    SENSOR_ROTATION_SPEED,
    SENSOR_VOLTAGE,
    SENSOR_CURRENT,
]

OPERATIONAL_COLUMNS = [
    OPERATIONAL_SETTING_1,
    OPERATIONAL_SETTING_2,
    OPERATIONAL_SETTING_3,
]

# Additional sensor columns (simulating 21-sensor C-MAPSS style)
EXTENDED_SENSOR_PREFIX = "sensor_"
EXTENDED_SENSOR_COUNT = 21

# ── Model Names ───────────────────────────────────────────────────────────
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_XGBOOST = "xgboost"
MODEL_LIGHTGBM = "lightgbm"
MODEL_LSTM = "lstm"
MODEL_ISOLATION_FOREST = "isolation_forest"
MODEL_AUTOENCODER = "autoencoder"

# ── Metric Names ──────────────────────────────────────────────────────────
METRIC_ROC_AUC = "roc_auc"
METRIC_PR_AUC = "pr_auc"
METRIC_F1 = "f1_score"
METRIC_PRECISION = "precision"
METRIC_RECALL = "recall"
METRIC_RMSE = "rmse"
METRIC_MAE = "mae"
METRIC_R2 = "r2"
METRIC_COST_SCORE = "cost_score"

# ── Task Types ────────────────────────────────────────────────────────────
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASK_ANOMALY_DETECTION = "anomaly_detection"

# ── File Paths ────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FEATURES_DATA_DIR = "data/features"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# ── Business Constants ────────────────────────────────────────────────────
DEFAULT_FAILURE_WINDOW_CYCLES = 30
DEFAULT_DOWNTIME_COST_PER_HOUR = 10_000.0
DEFAULT_MAINTENANCE_COST = 2_000.0
DEFAULT_FALSE_NEGATIVE_MULTIPLIER = 5.0
DEFAULT_FALSE_POSITIVE_MULTIPLIER = 1.0
