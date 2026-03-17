"""
config.py — Central configuration for the repository

This repo is now positioned as a battery/BMS-first ML monitoring platform built
on top of a reusable monitoring core. Some generic tabular/churn defaults are
still present for legacy example paths and should be treated as reference
settings rather than the primary project identity.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MLFLOW_DIR = BASE_DIR / "mlruns"

for _dir in [DATA_DIR, MODELS_DIR, REPORTS_DIR, MLFLOW_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Drift Detection Thresholds ─────────────────────────────────────────────────
@dataclass
class DriftConfig:
    # Population Stability Index thresholds
    psi_warning:  float = 0.10   # 0.10–0.20: moderate shift
    psi_critical: float = 0.20   # >0.20:     significant shift

    # Kolmogorov–Smirnov test
    ks_p_value_threshold: float = 0.05

    # Jensen–Shannon Divergence
    jsd_warning:  float = 0.05
    jsd_critical: float = 0.10

    # Wasserstein distance (normalized)
    wasserstein_warning:  float = 0.10
    wasserstein_critical: float = 0.20

    # Chi-square for categorical features
    chi2_p_value_threshold: float = 0.05

    # Sliding window size (number of records per batch)
    window_size: int = 500

    # Minimum samples required for drift test
    min_samples: int = 100


# ── Model Performance Thresholds ───────────────────────────────────────────────
@dataclass
class PerformanceConfig:
    # Minimum acceptable AUC-ROC on production batches
    auc_warning:  float = 0.80
    auc_critical: float = 0.70

    # F1 score thresholds
    f1_warning:  float = 0.75
    f1_critical: float = 0.65

    # Precision / recall lower bounds
    precision_warning: float = 0.70
    recall_warning:    float = 0.65

    # Prediction drift (distribution of output scores)
    pred_drift_warning:  float = 0.08
    pred_drift_critical: float = 0.15


# ── Model Training Config ──────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    n_estimators:   int   = 300
    max_depth:      int   = 6
    learning_rate:  float = 0.05
    subsample:      float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 2.9  # n_neg/n_pos for ~26% churn: 74/26 ≈ 2.85
    random_state:   int   = 42
    test_size:      float = 0.20
    val_size:       float = 0.10

    # Feature groups for structured monitoring
    numerical_features: List[str] = field(default_factory=lambda: [
        "tenure", "monthly_charges", "total_charges",
        "num_products", "num_support_calls",
        "avg_call_duration", "days_since_last_login",
        "data_usage_gb", "billing_amount_variance",
        "customer_lifetime_value"
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "contract_type", "payment_method",
        "internet_service", "tech_support",
        "online_security", "paperless_billing"
    ])
    target: str = "churn"


# ── Alert Config ────────────────────────────────────────────────────────────────
@dataclass
class AlertConfig:
    # Webhook URLs (set via .env or env vars)
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    email_smtp_host:   str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    email_smtp_port:   int = int(os.getenv("SMTP_PORT", "587"))
    email_sender:      str = os.getenv("ALERT_EMAIL_SENDER", "")
    email_recipients:  List[str] = field(default_factory=list)

    # Cooldown: don't re-alert for same issue within N minutes
    alert_cooldown_minutes: int = 30

    # Severity levels
    SEVERITY_INFO     = "INFO"
    SEVERITY_WARNING  = "WARNING"
    SEVERITY_CRITICAL = "CRITICAL"


# ── MLflow Config ──────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "battery_ml_monitoring_platform"

# Path.as_uri() produces a valid RFC-3986 URI on every platform:
#   Linux/macOS  /home/user/mlruns  ->  file:///home/user/mlruns
#   Windows      C:/Users/mlruns    ->  file:///C:/Users/mlruns  (correct)
# This fixes the Windows error: 'file://C:\...' is not a valid remote uri.
MLFLOW_TRACKING_URI = MLFLOW_DIR.as_uri()

# ── Dashboard Config ───────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = False

# ── API Server Config ───────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    """Configuration for the FastAPI real-time inference server."""
    host:               str  = "0.0.0.0"
    port:               int  = 8000
    workers:            int  = 1       # uvicorn worker count (1 for dev, 4+ for prod)
    reload:             bool = False   # auto-reload on code change (dev only)
    window_size:        int  = 1_000   # max records in sliding window
    drift_check_every:  int  = 500     # run drift check after N predictions
    default_threshold:  float = 0.50   # classification threshold


# ── ROI / Business Impact Config ───────────────────────────────────────────────
@dataclass
class ROIConfig:
    """
    Business parameters used by the ROI Calculator to translate AUC drops
    into estimated revenue loss and wasted outreach spend.

    Tune these to match the real company's numbers before deploying.
    """
    # Total customers scored by the model each month
    monthly_customers: int   = 5_000

    # Average customer lifetime value in local currency (₹ or $)
    avg_customer_ltv: float  = 3_600

    # Cost of one outreach call (agent time + tooling)
    outreach_cost_per_call: float = 50.0

    # True churn rate — fraction of customers who churn each month
    true_churn_rate: float   = 0.10

    # Fraction of flagged customers the retention team actually contacts
    outreach_coverage: float = 0.70

    # Estimated fraction of contacted churners successfully retained
    retention_success_rate: float = 0.30

    # Recall sensitivity factor: recall_drop ≈ auc_drop × this value
    # Calibrated empirically for XGBoost on imbalanced binary tasks
    recall_sensitivity: float = 1.8

    # Compute cost of one full retraining run (cloud/on-prem cost in ₹/$)
    # Set to 0 if running locally with negligible marginal cost
    retrain_compute_cost: float = 0.0


# ── Registry / Multi-Model Config ──────────────────────────────────────────────
@dataclass
class RegistryConfig:
    """
    Configuration for the multi-model registry.

    Defaults to a local SQLite database so the project runs zero-config.
    Switch db_url to a PostgreSQL connection string for production.
    """
    # Database connection URL.
    # SQLite  (default, zero-config): leave as None → uses models/registry.db
    # PostgreSQL (production):        "postgresql://user:pass@host:5432/mlmonitor"
    db_url: Optional[str] = None

    # Export fleet health JSON for the dashboard after each monitoring run
    export_fleet_json: bool = True

    # Path for the exported fleet health JSON (used by dashboard)
    fleet_json_filename: str = "fleet_health.json"

    # How many batches to keep in batch_metrics per model before archiving
    # 0 = keep all (default, fine for SQLite; set a limit for large Postgres deployments)
    max_batches_per_model: int = 0

    # Model names to include in the multi-model monitoring demo
    # These must match names registered via db_migrate.py
    demo_model_names: List[str] = field(default_factory=lambda: [
        "churn_xgboost",
        "ltv_xgboost",
        "propensity_xgb",
    ])




# ── Singleton instances ─────────────────────────────────────────────────────────
drift_config       = DriftConfig()
performance_config = PerformanceConfig()
model_config       = ModelConfig()
alert_config       = AlertConfig()
api_config         = APIConfig()
roi_config         = ROIConfig()
registry_config    = RegistryConfig()
