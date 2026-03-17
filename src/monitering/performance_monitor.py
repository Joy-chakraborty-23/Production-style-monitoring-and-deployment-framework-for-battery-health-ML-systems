"""
performance_monitor.py — Model Performance Monitoring + Retraining Trigger

Tracks model metrics over production batches:
  - AUC-ROC, F1, Precision, Recall (requires ground truth labels)
  - Calibration drift (Brier score)
  - Prediction score distribution shift
  - Rolling window performance trend

Retraining is triggered when:
  1. AUC drops below critical threshold, OR
  2. Both drift severity is critical AND AUC drops below warning threshold
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import mlflow
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve

from src.config import (
    performance_config, drift_config,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR
)


@dataclass
class PerformanceBatch:
    batch_id:         int
    n_samples:        int
    auc_roc:          float
    avg_precision:    float
    f1:               float
    precision:        float
    recall:           float
    brier_score:      float
    churn_rate_true:  float
    churn_rate_pred:  float
    drift_severity:   str = "none"
    retrain_triggered: bool = False
    retrain_reason:   str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class PerformanceMonitor:
    """
    Tracks model performance per batch and manages the retraining trigger.

    Usage:
        monitor = PerformanceMonitor(pipeline)
        status = monitor.evaluate_batch(X_batch, y_batch, batch_id=5)
    """

    def __init__(self, pipeline, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        self.pipeline    = pipeline
        self.history: List[PerformanceBatch] = []
        self.experiment_name = experiment_name
        self._retrain_cooldown = 0   # batches remaining in cooldown

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

    def evaluate_batch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        batch_id: int,
        drift_severity: str = "none",
    ) -> PerformanceBatch:
        """
        Score a production batch and evaluate model performance.

        Args:
            X:              Feature dataframe
            y:              Ground-truth labels
            batch_id:       Sequential batch ID
            drift_severity: Overall drift severity from DriftDetector

        Returns:
            PerformanceBatch with all metrics
        """
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_arr  = y.values if hasattr(y, "values") else np.array(y)

        # Guard against degenerate batches
        if len(np.unique(y_arr)) < 2:
            logger.warning(f"Batch {batch_id}: only one class present — skipping AUC")
            auc = float("nan")
        else:
            auc = roc_auc_score(y_arr, y_prob)

        batch = PerformanceBatch(
            batch_id=batch_id,
            n_samples=len(X),
            auc_roc=round(float(auc) if not np.isnan(auc) else 0.0, 4),
            avg_precision=round(float(average_precision_score(y_arr, y_prob)), 4),
            f1=round(float(f1_score(y_arr, y_pred, zero_division=0)), 4),
            precision=round(float(precision_score(y_arr, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_arr, y_pred, zero_division=0)), 4),
            brier_score=round(float(brier_score_loss(y_arr, y_prob)), 4),
            churn_rate_true=round(float(y_arr.mean()), 4),
            churn_rate_pred=round(float(y_pred.mean()), 4),
            drift_severity=drift_severity,
        )

        # ── Retraining trigger ────────────────────────────────────────────
        batch.retrain_triggered, batch.retrain_reason = self._check_retrain(batch)

        self.history.append(batch)
        self._log_to_mlflow(batch)

        level = "WARNING" if batch.retrain_triggered else "INFO"
        logger.log(
            level,
            f"Batch {batch_id}: AUC={batch.auc_roc:.4f}  F1={batch.f1:.4f}  "
            f"Drift={drift_severity}  Retrain={batch.retrain_triggered}"
        )
        return batch

    def _check_retrain(self, batch: PerformanceBatch) -> tuple[bool, str]:
        """
        Composite retraining trigger logic:
          - AUC below critical threshold → immediate retrain
          - AUC below warning AND drift is critical → retrain
          - AUC declining trend (3-batch rolling drop > 5%) → retrain
        """
        if self._retrain_cooldown > 0:
            self._retrain_cooldown -= 1
            return False, ""

        # Rule 1: Hard AUC floor
        if batch.auc_roc < performance_config.auc_critical:
            self._retrain_cooldown = 3
            return True, f"AUC {batch.auc_roc:.4f} below critical threshold {performance_config.auc_critical}"

        # Rule 2: Drift + degradation combo
        if (
            batch.drift_severity in ("warning", "critical")
            and batch.auc_roc < performance_config.auc_warning
        ):
            self._retrain_cooldown = 3
            return True, (
                f"AUC {batch.auc_roc:.4f} below warning threshold "
                f"with {batch.drift_severity} drift detected"
            )

        # Rule 3: Rolling trend (need 3 batches)
        if len(self.history) >= 3:
            recent_aucs = [b.auc_roc for b in self.history[-3:]]
            trend_drop  = recent_aucs[0] - batch.auc_roc
            if trend_drop > 0.05 and batch.auc_roc < performance_config.auc_warning:
                self._retrain_cooldown = 3
                return True, (
                    f"Rolling AUC drop: {recent_aucs[0]:.4f}→{batch.auc_roc:.4f} "
                    f"({trend_drop:.3f} over 3 batches)"
                )

        return False, ""

    def _log_to_mlflow(self, batch: PerformanceBatch) -> None:
        """Log batch metrics to MLflow as a monitoring run."""
        with mlflow.start_run(run_name=f"monitor_batch_{batch.batch_id:03d}", nested=True):
            mlflow.log_metrics({
                "auc_roc":         batch.auc_roc,
                "avg_precision":   batch.avg_precision,
                "f1":              batch.f1,
                "precision":       batch.precision,
                "recall":          batch.recall,
                "brier_score":     batch.brier_score,
                "churn_rate_true": batch.churn_rate_true,
                "churn_rate_pred": batch.churn_rate_pred,
            }, step=batch.batch_id)
            mlflow.log_params({
                "batch_id":         batch.batch_id,
                "n_samples":        batch.n_samples,
                "drift_severity":   batch.drift_severity,
                "retrain_triggered": int(batch.retrain_triggered),
            })
            if batch.retrain_triggered:
                mlflow.set_tag("retrain_trigger", batch.retrain_reason)

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame([b.to_dict() for b in self.history])

    def get_rolling_auc(self, window: int = 3) -> pd.Series:
        df = self.get_history_df()
        return df["auc_roc"].rolling(window).mean()

    def save_history(self, path: Path = None) -> None:
        if path is None:
            path = MODELS_DIR / "performance_history.json"
        df = self.get_history_df()
        df.to_json(path, orient="records", indent=2)
        logger.info(f"Performance history saved to {path}")

    def load_history(self, path: Path = None) -> None:
        if path is None:
            path = MODELS_DIR / "performance_history.json"
        if not path.exists():
            logger.warning(f"No history file at {path}")
            return
        df = pd.read_json(path)
        self.history = [PerformanceBatch(**row) for row in df.to_dict("records")]
        logger.info(f"Loaded {len(self.history)} historical batches")
