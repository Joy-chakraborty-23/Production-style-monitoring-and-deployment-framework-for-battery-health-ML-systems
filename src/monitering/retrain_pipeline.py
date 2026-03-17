"""
retrain_pipeline.py — Automated Retraining Pipeline

When triggered, this pipeline:
  1. Loads recent production data + original reference data
  2. Retrains the XGBoost pipeline from scratch
  3. Evaluates new model vs current model on holdout
  4. Promotes new model only if it improves AUC by > min_improvement
  5. Logs everything to MLflow with comparison tags
  6. Returns a RetrainingResult with the decision
"""

import joblib
import mlflow
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
from sklearn.metrics import roc_auc_score

from src.config import (
    model_config, MODELS_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from src.models.train_model import build_model_pipeline, compute_metrics
from src.data.data_generator import prepare_train_test_split


MIN_IMPROVEMENT_AUC = 0.01   # must beat current model by this margin


@dataclass
class RetrainingResult:
    triggered_at_batch: int
    trigger_reason:     str
    old_auc:            float
    new_auc:            float
    promoted:           bool
    promotion_reason:   str
    run_id:             str
    model_path:         Path


class RetrainingPipeline:
    """
    Manages model retraining lifecycle.

    Usage:
        retrain = RetrainingPipeline(current_pipeline)
        result = retrain.run(production_batches, batch_id=9, reason="AUC below threshold")
    """

    def __init__(self, current_pipeline, holdout_X: pd.DataFrame, holdout_y: pd.Series):
        self.current_pipeline = current_pipeline
        self.holdout_X = holdout_X
        self.holdout_y = holdout_y
        self.retraining_history: list[RetrainingResult] = []

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def _current_holdout_auc(self) -> float:
        try:
            y_prob = self.current_pipeline.predict_proba(self.holdout_X)[:, 1]
            return float(roc_auc_score(self.holdout_y, y_prob))
        except Exception:
            return 0.0

    def run(
        self,
        new_data_batches: list[pd.DataFrame],
        batch_id: int,
        trigger_reason: str,
    ) -> RetrainingResult:
        """
        Execute retraining on accumulated production data.

        Args:
            new_data_batches: List of recent production DataFrames
            batch_id:         Batch that triggered retraining
            trigger_reason:   Human-readable reason string

        Returns:
            RetrainingResult with promotion decision
        """
        logger.info(f"=== Retraining triggered at batch {batch_id} ===")
        logger.info(f"Reason: {trigger_reason}")

        # ── Build retraining dataset ────────────────────────────────────
        combined = pd.concat(new_data_batches, ignore_index=True)
        feature_cols = model_config.numerical_features + model_config.categorical_features

        # Filter to rows that have labels
        labeled = combined[combined[model_config.target].notna()].copy()
        logger.info(f"Retraining on {len(labeled)} labeled records from {len(new_data_batches)} batches")

        X_train, X_val, y_train, y_val = prepare_train_test_split(labeled)

        old_auc = self._current_holdout_auc()
        logger.info(f"Current model holdout AUC: {old_auc:.4f}")

        # ── Train new model ─────────────────────────────────────────────
        with mlflow.start_run(run_name=f"retrain_batch_{batch_id:03d}") as run:
            run_id = run.info.run_id

            new_pipeline = build_model_pipeline()
            new_pipeline.fit(X_train, y_train)

            y_prob_val = new_pipeline.predict_proba(X_val)[:, 1]
            y_pred_val = (y_prob_val >= 0.5).astype(int)
            val_metrics = compute_metrics(y_val.values, y_pred_val, y_prob_val)

            # Holdout evaluation
            y_prob_holdout = new_pipeline.predict_proba(self.holdout_X)[:, 1]
            new_auc = float(roc_auc_score(self.holdout_y, y_prob_holdout))

            mlflow.log_params({
                "trigger_batch":      batch_id,
                "trigger_reason":     trigger_reason,
                "n_retrain_samples":  len(labeled),
                "n_batches_included": len(new_data_batches),
            })
            mlflow.log_metrics({
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "holdout_auc_old": round(old_auc, 4),
                "holdout_auc_new": round(new_auc, 4),
                "auc_improvement":  round(new_auc - old_auc, 4),
            })

            # ── Promotion decision ──────────────────────────────────────
            improvement = new_auc - old_auc
            if improvement >= MIN_IMPROVEMENT_AUC:
                promoted = True
                reason = f"New model AUC {new_auc:.4f} beats old {old_auc:.4f} by {improvement:.4f}"
                mlflow.set_tag("model_status", "promoted")
                self._save_new_model(new_pipeline)
                self.current_pipeline = new_pipeline
            else:
                promoted = False
                reason = (
                    f"New model AUC {new_auc:.4f} did not improve over {old_auc:.4f} "
                    f"(improvement={improvement:.4f} < {MIN_IMPROVEMENT_AUC})"
                )
                mlflow.set_tag("model_status", "rejected")

            logger.info(f"Promotion: {promoted} — {reason}")
            mlflow.set_tag("promoted", str(promoted))
            mlflow.log_text(reason, "promotion_decision.txt")

        result = RetrainingResult(
            triggered_at_batch=batch_id,
            trigger_reason=trigger_reason,
            old_auc=old_auc,
            new_auc=new_auc,
            promoted=promoted,
            promotion_reason=reason,
            run_id=run_id,
            model_path=MODELS_DIR / "churn_pipeline.joblib",
        )
        self.retraining_history.append(result)
        return result

    def _save_new_model(self, pipeline) -> None:
        model_path = MODELS_DIR / "churn_pipeline.joblib"
        # Archive old model
        archive_path = MODELS_DIR / f"churn_pipeline_prev.joblib"
        if model_path.exists():
            import shutil
            shutil.copy(model_path, archive_path)
        joblib.dump(pipeline, model_path)
        logger.info(f"New model saved to {model_path}")
