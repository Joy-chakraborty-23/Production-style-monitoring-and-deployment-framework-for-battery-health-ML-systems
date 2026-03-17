"""
scripts/db_migrate.py — Database Schema Migration & Multi-Model Seeder

Creates the full registry schema and registers the three model variants
that the multi-model demo monitors simultaneously:
  1. churn_xgboost   — primary churn prediction model (already trained)
  2. ltv_xgboost     — customer lifetime value segment model (simulated)
  3. propensity_xgb  — upsell propensity model (simulated)

Run once before starting the multi-model monitoring loop:
    python scripts/db_migrate.py

For PostgreSQL production:
    python scripts/db_migrate.py --db postgresql://user:pass@host:5432/mlmonitor

Schema created:
  models        (model registry)
  batch_metrics (per-model per-batch time-series)
  retrain_log   (retraining events)
  drift_details (per-feature drift scores)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
from src.monitoring.model_registry import ModelRegistry
from src.config import MODELS_DIR


def migrate(db_url: str = None) -> ModelRegistry:
    """
    Initialise schema and seed the three demo models.
    Returns the populated registry.
    """
    registry = ModelRegistry(db_url)

    logger.info("Running schema migration...")

    # ── Register the three demo models ────────────────────────────────────
    models_to_register = [
        {
            "name":          "churn_xgboost",
            "version":       "1.0.0",
            "description":   "Customer churn prediction — primary retention model",
            "pipeline_path": str(MODELS_DIR / "churn_pipeline.joblib"),
            "thresholds": {
                "auc_warning":  0.80,
                "auc_critical": 0.70,
                "f1_warning":   0.75,
                "psi_warning":  0.10,
                "psi_critical": 0.20,
            },
            "tags": {
                "domain":   "retention",
                "team":     "ds-platform",
                "priority": "high",
            },
        },
        {
            "name":          "ltv_xgboost",
            "version":       "1.0.0",
            "description":   "Customer LTV segmentation — identifies high-value at-risk customers",
            "pipeline_path": str(MODELS_DIR / "ltv_pipeline.joblib"),
            "thresholds": {
                "auc_warning":  0.78,   # slightly looser — LTV is harder to predict
                "auc_critical": 0.68,
                "f1_warning":   0.72,
                "psi_warning":  0.12,
                "psi_critical": 0.22,
            },
            "tags": {
                "domain":   "finance",
                "team":     "growth",
                "priority": "medium",
            },
        },
        {
            "name":          "propensity_xgb",
            "version":       "1.0.0",
            "description":   "Upsell propensity — scores customers likely to upgrade their plan",
            "pipeline_path": str(MODELS_DIR / "propensity_pipeline.joblib"),
            "thresholds": {
                "auc_warning":  0.76,
                "auc_critical": 0.66,
                "f1_warning":   0.70,
                "psi_warning":  0.10,
                "psi_critical": 0.20,
            },
            "tags": {
                "domain":   "growth",
                "team":     "marketing",
                "priority": "medium",
            },
        },
    ]

    ids = {}
    for m in models_to_register:
        mid = registry.register_model(**m)
        ids[m["name"]] = mid
        logger.info(f"  Registered: {m['name']} v{m['version']}  →  id={mid}")

    logger.success(f"Migration complete. {len(ids)} models registered.")
    logger.info(f"Registry location: {registry.db_url}")
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Monitor DB Migration")
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "Database URL. "
            "Default: SQLite at models/registry.db  "
            "Production: postgresql://user:pass@host:5432/mlmonitor"
        ),
    )
    args = parser.parse_args()
    reg = migrate(args.db)

    print("\nCurrent fleet:")
    print(reg.list_models()[["model_id", "name", "version", "status", "description"]].to_string(index=False))
