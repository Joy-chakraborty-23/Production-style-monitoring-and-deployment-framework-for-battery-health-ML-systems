"""
monitoring_loop.py — Main Orchestration Loop

Simulates a real-time production monitoring system:
  1. Load reference data + trained model
  2. For each production batch:
     a. Detect feature drift (PSI, KS, JSD, Wasserstein, Chi2)
     b. Evaluate model performance (AUC, F1, precision, recall)
     c. Fire alerts if thresholds exceeded
     d. Trigger retraining if conditions met
     e. Log everything to MLflow
  3. Save monitoring history for the dashboard

Run this as the main entry point:
    python -m src.monitoring_loop
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

# ── Sys path fix for running as module ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    performance_config, drift_config, roi_config, registry_config,
    DATA_DIR, MODELS_DIR
)
from src.data.data_generator import (
    generate_reference_data, generate_production_batch,
    save_datasets, prepare_train_test_split
)
from src.models.train_model import train_and_log, load_model
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.retrain_pipeline import RetrainingPipeline
from src.monitoring.roi_calculator import ROICalculator
from src.monitoring.model_registry import ModelRegistry
from src.alerts.alert_manager import (
    AlertManager, fire_drift_alert, fire_performance_alert, fire_retrain_alert
)


BATCH_CONFIGS = [
    # (batch_id, drift_type, drift_intensity, label="scenario description")
    (0,  None,        0.0, "Stable baseline"),
    (1,  None,        0.0, "Stable baseline"),
    (2,  None,        0.0, "Stable baseline"),
    (3,  None,        0.0, "Stable baseline"),
    (4,  "gradual",   0.3, "Gradual drift begins"),
    (5,  "gradual",   0.5, "Gradual drift accelerates"),
    (6,  "gradual",   0.6, "Moderate covariate shift"),
    (7,  "gradual",   0.7, "Significant covariate shift"),
    (8,  "sudden",    0.9, "SUDDEN concept drift — market event"),
    (9,  "covariate", 0.8, "Sustained covariate shift"),
    (10, "covariate", 0.7, "Sustained covariate shift"),
    (11, "covariate", 0.6, "Shift begins recovering"),
    (12, "covariate", 0.3, "Partial recovery"),
    (13, None,        0.0, "Post-retrain stable"),
    (14, None,        0.0, "Post-retrain stable"),
]


def run_monitoring_loop(
    n_samples_per_batch: int = 500,
    simulate_delay: float = 0.0,   # seconds between batches (0 = instant for demo)
) -> dict:
    """
    Full end-to-end monitoring simulation.

    Returns:
        dict with performance_history, drift_reports, alerts_summary
    """
    logger.info("=" * 60)
    logger.info("  ML Monitoring System — Starting")
    logger.info("=" * 60)

    # ── Step 1: Generate / load reference data ────────────────────────────
    ref_path = DATA_DIR / "reference_data.csv"
    if ref_path.exists():
        logger.info(f"Loading reference data from {ref_path}")
        df_ref = pd.read_csv(ref_path)
    else:
        logger.info("Generating reference data...")
        df_ref = generate_reference_data(n_samples=5000)
        df_ref.to_csv(ref_path, index=False)

    # ── Step 2: Train model (or load existing) ────────────────────────────
    model_path = MODELS_DIR / "churn_pipeline.joblib"
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        pipeline = load_model(model_path)
    else:
        logger.info("Training initial model...")
        result   = train_and_log(df_ref)
        pipeline = result["pipeline"]

    # ── Step 3: Build holdout for retraining evaluation ───────────────────
    _, X_holdout, _, y_holdout = prepare_train_test_split(df_ref)

    # ── Step 4: Initialize monitoring components ──────────────────────────
    detector  = DriftDetector(df_ref)
    monitor   = PerformanceMonitor(pipeline)
    retrainer = RetrainingPipeline(pipeline, X_holdout, y_holdout)
    alert_mgr = AlertManager()

    # ── ROI Calculator — wire business params from config ─────────────────
    # Baseline AUC is captured from the trained model's test metrics so the
    # calculator always compares against the actual deployment-day performance.
    try:
        _ref_X = df_ref[[c for c in df_ref.columns
                          if c not in ["churn", "batch_id", "drift_type"]]].copy()
        _ref_y = df_ref["churn"]
        from sklearn.metrics import roc_auc_score
        _baseline_auc = float(roc_auc_score(
            _ref_y, pipeline.predict_proba(_ref_X)[:, 1]
        ))
    except Exception:
        _baseline_auc = 0.87   # safe fallback

    roi_calc = ROICalculator(
        monthly_customers      = roi_config.monthly_customers,
        avg_ltv                = roi_config.avg_customer_ltv,
        outreach_cost_per_call = roi_config.outreach_cost_per_call,
        baseline_auc           = _baseline_auc,
        true_churn_rate        = roi_config.true_churn_rate,
        outreach_coverage      = roi_config.outreach_coverage,
        recall_sensitivity     = roi_config.recall_sensitivity,
    )
    logger.info(f"ROI Calculator initialised — baseline AUC: {_baseline_auc:.4f}")

    # ── Step 4b: Initialise Multi-Model Registry ──────────────────────────
    registry = ModelRegistry(registry_config.db_url)
    _model_id = registry.register_model(
        name          = "churn_xgboost",
        version       = "1.0.0",
        description   = "Customer churn prediction — XGBoost pipeline",
        pipeline_path = str(MODELS_DIR / "churn_pipeline.joblib"),
        thresholds    = {
            "auc_warning":  performance_config.auc_warning,
            "auc_critical": performance_config.auc_critical,
            "psi_warning":  drift_config.psi_warning,
            "psi_critical": drift_config.psi_critical,
        },
        tags = {"domain": "retention", "team": "ds-platform"},
    )
    logger.info(f"Registry: churn_xgboost registered (model_id={_model_id})")

    # Reference predictions for prediction drift comparison
    ref_feature_cols = [c for c in df_ref.columns
                        if c not in ["churn", "batch_id", "drift_type"]]
    ref_X = df_ref[ref_feature_cols].copy()
    ref_preds = pipeline.predict_proba(ref_X)[:, 1]

    # ── Step 5: Main batch loop ───────────────────────────────────────────
    all_drift_reports  = []
    accumulated_batches = [df_ref]    # for retraining
    batch_summaries    = []

    for batch_id, drift_type, drift_intensity, scenario in BATCH_CONFIGS:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing Batch {batch_id:03d}: {scenario}")

        # Generate batch
        batch_df = generate_production_batch(
            n_samples=n_samples_per_batch,
            batch_id=batch_id,
            drift_type=drift_type,
            drift_intensity=drift_intensity,
        )
        accumulated_batches.append(batch_df)

        # Feature columns only (no target for drift detection)
        feature_cols = [c for c in batch_df.columns
                        if c not in ["churn", "batch_id", "drift_type"]]
        X_batch = batch_df[feature_cols].copy()
        y_batch = batch_df["churn"]

        # ── Drift Detection ────────────────────────────────────────────
        prod_preds = pipeline.predict_proba(X_batch)[:, 1]
        drift_report = detector.detect(
            batch_df,
            batch_id=batch_id,
            ref_predictions=ref_preds,
            prod_predictions=prod_preds,
        )
        all_drift_reports.append(drift_report)

        # Fire drift alert if needed
        if drift_report.overall_severity in ("warning", "critical"):
            fire_drift_alert(
                alert_mgr,
                batch_id=batch_id,
                severity=drift_report.overall_severity,
                drifted_features=drift_report.drifted_features,
                drift_score=drift_report.overall_drift_score,
            )

        # ── Performance Monitoring ────────────────────────────────────
        perf_batch = monitor.evaluate_batch(
            X_batch, y_batch,
            batch_id=batch_id,
            drift_severity=drift_report.overall_severity,
        )

        # Fire performance alert
        if perf_batch.auc_roc < performance_config.auc_critical:
            fire_performance_alert(
                alert_mgr, batch_id, "CRITICAL",
                perf_batch.auc_roc, perf_batch.f1
            )
        elif perf_batch.auc_roc < performance_config.auc_warning:
            fire_performance_alert(
                alert_mgr, batch_id, "WARNING",
                perf_batch.auc_roc, perf_batch.f1
            )

        # ── Retraining Trigger ────────────────────────────────────────
        if perf_batch.retrain_triggered:
            logger.warning(f"Retraining triggered: {perf_batch.retrain_reason}")

            # Compute NPV of the retrain before running it
            _roi_retrain = roi_calc.compute_retrain_value(
                current_auc=perf_batch.auc_roc,
                expected_new_auc=min(perf_batch.auc_roc + 0.06, roi_calc.baseline_auc),
                compute_cost=roi_config.retrain_compute_cost,
            )
            logger.info(f"Retrain ROI: {_roi_retrain['rationale']}")

            retrain_result = retrainer.run(
                new_data_batches=accumulated_batches[-8:],   # last 8 batches
                batch_id=batch_id,
                trigger_reason=perf_batch.retrain_reason,
            )
            fire_retrain_alert(
                alert_mgr, batch_id,
                retrain_result.promoted,
                retrain_result.old_auc,
                retrain_result.new_auc,
                retrain_result.promotion_reason,
            )
            registry.log_retrain_event(
                model_id         = _model_id,
                trigger_batch    = batch_id,
                trigger_reason   = perf_batch.retrain_reason,
                old_auc          = retrain_result.old_auc,
                new_auc          = retrain_result.new_auc,
                promoted         = retrain_result.promoted,
                promotion_reason = retrain_result.promotion_reason,
                mlflow_run_id    = retrain_result.run_id,
            )
            if retrain_result.promoted:
                # Update components with new model
                pipeline = retrainer.current_pipeline
                monitor.pipeline = pipeline
                ref_preds = pipeline.predict_proba(ref_X)[:, 1]

        # ── ROI Impact ────────────────────────────────────────────────
        roi_result = roi_calc.compute_monthly_loss(
            current_auc=perf_batch.auc_roc,
            current_f1=perf_batch.f1,
        )
        if roi_result.auc_drop > 0:
            logger.log(
                "WARNING" if roi_result.status != "healthy" else "INFO",
                roi_result.headline,
            )

        # ── Registry: persist this batch ──────────────────────────────
        registry.log_batch_metrics(
            model_id           = _model_id,
            batch_id           = batch_id,
            auc_roc            = perf_batch.auc_roc,
            f1_score           = perf_batch.f1,
            precision_val      = perf_batch.precision,
            recall_val         = perf_batch.recall,
            brier_score        = perf_batch.brier_score,
            drift_score        = drift_report.overall_drift_score,
            drift_severity     = drift_report.overall_severity,
            n_drifted_feats    = drift_report.n_drifted_features,
            n_samples          = len(X_batch),
            retrain_triggered  = perf_batch.retrain_triggered,
            roi_monthly_loss   = roi_result.total_monthly_loss,
            roi_missed_churners= roi_result.missed_churners,
        )
        registry.log_drift_details(
            model_id    = _model_id,
            batch_id    = batch_id,
            drift_results = drift_report.drift_results,
        )

        # ── Summary ───────────────────────────────────────────────────
        batch_summaries.append({
            "batch_id":              batch_id,
            "scenario":              scenario,
            "drift_type":            drift_type or "none",
            "drift_severity":        drift_report.overall_severity,
            "n_drifted_feats":       drift_report.n_drifted_features,
            "drift_score":           round(drift_report.overall_drift_score, 4),
            "auc_roc":               perf_batch.auc_roc,
            "f1":                    perf_batch.f1,
            "precision":             perf_batch.precision,
            "recall":                perf_batch.recall,
            "churn_rate_true":       perf_batch.churn_rate_true,
            "retrain_triggered":     perf_batch.retrain_triggered,
            # ── ROI fields (new) ──────────────────────────────────────
            "roi_auc_drop":          roi_result.auc_drop,
            "roi_missed_churners":   roi_result.missed_churners,
            "roi_revenue_at_risk":   roi_result.revenue_at_risk,
            "roi_wasted_outreach":   roi_result.wasted_outreach,
            "roi_total_monthly_loss":roi_result.total_monthly_loss,
            "roi_total_annual_loss": roi_result.total_annual_loss,
            "roi_status":            roi_result.status,
            "roi_headline":          roi_result.headline,
        })

        if simulate_delay > 0:
            time.sleep(simulate_delay)

    # ── Step 6: Save all artifacts ────────────────────────────────────────
    monitor.save_history()

    # Export registry fleet health for the dashboard
    if registry_config.export_fleet_json:
        registry.export_fleet_json(MODELS_DIR / registry_config.fleet_json_filename)

    summary_df = pd.DataFrame(batch_summaries)
    summary_path = MODELS_DIR / "monitoring_summary.json"
    summary_df.to_json(summary_path, orient="records", indent=2)

    # Save drift report details
    drift_details = []
    for report in all_drift_reports:
        for result in report.drift_results:
            d = result.to_dict()
            d["batch_id"] = report.batch_id
            drift_details.append(d)
    pd.DataFrame(drift_details).to_json(
        MODELS_DIR / "drift_details.json", orient="records", indent=2
    )

    # Save retraining history
    if retrainer.retraining_history:
        retrain_log = [
            {
                "triggered_at_batch": r.triggered_at_batch,
                "trigger_reason":     r.trigger_reason,
                "old_auc":            r.old_auc,
                "new_auc":            r.new_auc,
                "promoted":           r.promoted,
                "promotion_reason":   r.promotion_reason,
                "run_id":             r.run_id,
            }
            for r in retrainer.retraining_history
        ]
        with open(MODELS_DIR / "retrain_log.json", "w") as f:
            json.dump(retrain_log, f, indent=2)

    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Monitoring Loop Complete")
    logger.info("=" * 60)
    logger.info(f"  Batches processed: {len(batch_summaries)}")
    logger.info(f"  Alerts fired:      {alert_mgr.get_summary()}")
    logger.info(f"  Retraining runs:   {len(retrainer.retraining_history)}")
    if retrainer.retraining_history:
        final_auc = retrainer.retraining_history[-1].new_auc
        logger.info(f"  Final model AUC:   {final_auc:.4f}")

    return {
        "batch_summaries":    summary_df,
        "performance_history": monitor.get_history_df(),
        "alert_summary":      alert_mgr.get_summary(),
        "n_retraining_runs":  len(retrainer.retraining_history),
    }


if __name__ == "__main__":
    results = run_monitoring_loop()
    print("\nFinal batch summary:")
    print(results["batch_summaries"][
        ["batch_id", "scenario", "drift_severity", "auc_roc", "f1", "retrain_triggered"]
    ].to_string(index=False))
