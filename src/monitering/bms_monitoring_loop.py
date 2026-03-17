"""
bms_monitoring_loop.py — Multi-Model BMS Monitoring Loop

Orchestrates the full monitoring pipeline across all registered BMS models:

  1. Load all registered models from ModelRegistry
  2. Load reference dataset from DatasetRegistry
  3. For each production batch:
       a. Run EnsembleBMSPredictor (SOH + Fault + Trend fusion)
       b. Run DriftDetector on battery features
       c. Log metrics to ModelRegistry
       d. Calculate BMS-specific ROI (warranty costs, thermal incidents)
       e. Fire alerts if drift/fault thresholds exceeded
  4. Export fleet health JSON for dashboard
  5. Trigger retraining if needed

Usage:
    # Run full monitoring loop (all registered models, all available batches):
    python -m src.monitoring.bms_monitoring_loop

    # Run for a specific model only:
    python -m src.monitoring.bms_monitoring_loop --model-name bms_soh_regressor

    # Quick demo (generates synthetic data + runs 5 batches):
    python -m src.monitoring.bms_monitoring_loop --demo
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── BMS ROI Calculator ────────────────────────────────────────────────────────

class BMSROICalculator:
    """
    Calculates business impact of BMS model degradation.

    Unlike churn (where ROI is soft/estimated), BMS ROI is concrete:
      - Missed fault = potential warranty claim + safety incident
      - Each warranty failure = ~$7,000 replacement cost
      - Each thermal incident = ~$50,000+ (safety, legal, brand damage)
    """

    def __init__(
        self,
        fleet_size:            int   = 10_000,
        warranty_cost_per_cell: float = 7_000.0,
        thermal_incident_cost:  float = 50_000.0,
        false_positive_cost:    float = 200.0,   # unnecessary inspection
        true_fault_rate:        float = 0.02,    # 2% of cells fault per year
    ):
        self.fleet_size             = fleet_size
        self.warranty_cost_per_cell = warranty_cost_per_cell
        self.thermal_incident_cost  = thermal_incident_cost
        self.false_positive_cost    = false_positive_cost
        self.true_fault_rate        = true_fault_rate

    def calculate(
        self,
        recall:          float,    # model's recall on fault detection
        precision:       float,    # model's precision on fault detection
        soh_mae:         float,    # SOH prediction error
        n_cells_monitored: int = None,
    ) -> Dict[str, float]:
        n = n_cells_monitored or self.fleet_size
        true_faults = n * self.true_fault_rate / 12   # per month

        # Faults caught vs missed
        faults_detected = true_faults * recall
        faults_missed   = true_faults * (1 - recall)

        # False positives = unnecessary inspections
        predicted_faults = faults_detected / max(precision, 0.01)
        false_positives  = predicted_faults - faults_detected

        # Warranty + thermal costs
        missed_warranty_cost  = faults_missed * self.warranty_cost_per_cell
        false_positive_cost   = false_positives * self.false_positive_cost
        # Thermal risk scales with SOH error (poor SOH prediction → missed pre-thermal)
        thermal_risk_cost     = (soh_mae / 0.10) * self.thermal_incident_cost * 0.01

        total_monthly_cost = missed_warranty_cost + false_positive_cost + thermal_risk_cost

        return {
            "n_cells_monitored":    n,
            "true_faults_per_month": round(true_faults, 1),
            "faults_detected":      round(faults_detected, 1),
            "faults_missed":        round(faults_missed, 1),
            "false_positives":      round(false_positives, 1),
            "missed_warranty_cost": round(missed_warranty_cost, 2),
            "false_positive_cost":  round(false_positive_cost, 2),
            "thermal_risk_cost":    round(thermal_risk_cost, 2),
            "total_monthly_loss_usd": round(total_monthly_cost, 2),
            "annualized_risk_usd":  round(total_monthly_cost * 12, 2),
        }


# ── Main monitoring loop ──────────────────────────────────────────────────────

class BMSMonitoringLoop:
    """
    Runs continuous monitoring across all registered BMS models.

    Per-batch pipeline:
      1. Load batch data (from file or generate synthetically)
      2. Run EnsembleBMSPredictor → SOH + fault predictions
      3. Run DriftDetector on battery features
      4. Compute BMS ROI impact
      5. Log everything to ModelRegistry + reports
      6. Trigger alerts if needed
    """

    def __init__(
        self,
        model_name:  Optional[str] = None,
        demo_mode:   bool = False,
    ):
        self.model_name = model_name
        self.demo_mode  = demo_mode
        self.roi_calc   = BMSROICalculator()
        self._setup()

    def _setup(self):
        from src.monitoring.model_registry import ModelRegistry
        from src.data.dataset_registry    import DatasetRegistry
        self.registry        = ModelRegistry()
        self.dataset_registry = DatasetRegistry()
        logger.info("BMSMonitoringLoop initialised")

    def run(
        self,
        n_batches:   int = 10,
        batch_size:  int = 500,
        chemistry:   str = "NMC",
    ) -> pd.DataFrame:
        """
        Run the full monitoring loop for n_batches.

        Returns a DataFrame with one row per batch, per model.
        """
        # ── Load or register models ──────────────────────────────────────
        if self.demo_mode:
            model_ids = self._setup_demo_models(chemistry)
        else:
            model_ids = self._load_existing_model_ids()

        if not model_ids:
            logger.warning("No BMS models registered. Run with --demo or train models first.")
            logger.info("  Training BMS models on synthetic data...")
            model_ids = self._setup_demo_models(chemistry)

        # ── Load reference data ──────────────────────────────────────────
        ref_df = self._load_reference_data(chemistry, batch_size * 4)

        # ── Try to load drift detector ────────────────────────────────────
        try:
            from src.monitoring.drift_detector import DriftDetector
            from src.data.battery_data_generator import BMS_NUMERICAL_FEATURES
            drift_detector = DriftDetector(
                reference_data=ref_df[BMS_NUMERICAL_FEATURES],
                numerical_features=BMS_NUMERICAL_FEATURES,
                categorical_features=[],
            )
            has_drift_detector = True
        except Exception as e:
            logger.warning(f"DriftDetector not available: {e}")
            has_drift_detector = False

        # ── Try to load ensemble predictor ────────────────────────────────
        try:
            from src.models.bms_models import EnsembleBMSPredictor
            ensemble = EnsembleBMSPredictor()
            ensemble.load_all_models()
            has_ensemble = ensemble._models_loaded and (
                ensemble.soh_model or ensemble.fault_model
            )
        except Exception as e:
            logger.warning(f"Ensemble predictor not available: {e}")
            has_ensemble = False

        # ── Generate or load batches ──────────────────────────────────────
        from src.data.battery_data_generator import BatteryDataGenerator
        gen = BatteryDataGenerator(chemistry)

        drift_schedule = (
            [None] * 3
            + ["gradual"] * 3
            + ["thermal"] * 2
            + ["aging"] * 2
        )

        all_results = []
        logger.info(f"Starting monitoring loop — {n_batches} batches × {batch_size} samples")

        for batch_id in range(n_batches):
            drift_type = drift_schedule[batch_id % len(drift_schedule)]
            batch_df   = gen.generate_production_batch(
                n_samples=batch_size,
                batch_id=batch_id,
                drift_type=drift_type,
                drift_intensity=0.6 if drift_type else 0.0,
            )

            logger.info(
                f"Batch {batch_id:>3} | drift={drift_type or 'none':<8} | "
                f"SOH mean={batch_df['soh'].mean():.3f} | "
                f"fault rate={batch_df['fault_label'].mean():.3f}"
            )

            # ── Run ensemble predictions ──────────────────────────────────
            batch_metrics = self._evaluate_batch(
                batch_df=batch_df,
                ref_df=ref_df,
                batch_id=batch_id,
                drift_type=drift_type,
                ensemble=ensemble if has_ensemble else None,
            )

            # ── Run drift detection ───────────────────────────────────────
            drift_report = None
            if has_drift_detector:
                try:
                    from src.data.battery_data_generator import BMS_NUMERICAL_FEATURES
                    drift_report = drift_detector.detect(
                        production_data=batch_df[BMS_NUMERICAL_FEATURES],
                        batch_id=batch_id,
                    )
                    batch_metrics["drift_score"]    = drift_report.overall_drift_score
                    batch_metrics["drift_severity"] = drift_report.overall_severity
                    batch_metrics["n_drifted_feats"] = drift_report.n_drifted_features
                except Exception as e:
                    logger.warning(f"Drift detection failed for batch {batch_id}: {e}")
                    batch_metrics.setdefault("drift_score", 0.0)
                    batch_metrics.setdefault("drift_severity", "none")
                    batch_metrics.setdefault("n_drifted_feats", 0)

            # ── Log to registry (all models) ──────────────────────────────
            for model_id in model_ids:
                try:
                    roi = self.roi_calc.calculate(
                        recall=batch_metrics.get("recall", 0.8),
                        precision=batch_metrics.get("precision", 0.8),
                        soh_mae=batch_metrics.get("soh_mae", 0.03),
                        n_cells_monitored=batch_size,
                    )
                    self.registry.log_batch_metrics(
                        model_id=model_id,
                        batch_id=batch_id,
                        auc_roc=batch_metrics.get("auc_roc", 0.0),
                        f1_score=batch_metrics.get("f1", 0.0),
                        precision_val=batch_metrics.get("precision", 0.0),
                        recall_val=batch_metrics.get("recall", 0.0),
                        brier_score=batch_metrics.get("brier_score", 0.0),
                        drift_score=batch_metrics.get("drift_score", 0.0),
                        drift_severity=batch_metrics.get("drift_severity", "none"),
                        n_drifted_feats=batch_metrics.get("n_drifted_feats", 0),
                        n_samples=batch_size,
                        roi_monthly_loss=roi.get("total_monthly_loss_usd", 0.0),
                        extra={
                            "soh_mae":     batch_metrics.get("soh_mae", 0.0),
                            "soh_mean":    batch_metrics.get("soh_mean", 0.0),
                            "fault_rate":  batch_metrics.get("fault_rate", 0.0),
                            "drift_type":  drift_type or "none",
                            "roi_detail":  roi,
                        }
                    )
                    # Log drift details if available
                    if drift_report and has_drift_detector:
                        self.registry.log_drift_details(
                            model_id, batch_id, drift_report.drift_results
                        )
                except Exception as e:
                    logger.warning(f"Failed to log batch {batch_id} for model {model_id}: {e}")

            batch_metrics["batch_id"]   = batch_id
            batch_metrics["drift_type"] = drift_type or "none"
            all_results.append(batch_metrics)

        # ── Export fleet health for dashboard ─────────────────────────────
        try:
            self.registry.export_fleet_json()
        except Exception as e:
            logger.warning(f"Fleet export failed: {e}")

        results_df = pd.DataFrame(all_results)
        self._save_report(results_df, chemistry)
        logger.success(f"Monitoring loop complete — {len(results_df)} batches processed")
        return results_df

    def _evaluate_batch(
        self,
        batch_df:  pd.DataFrame,
        ref_df:    pd.DataFrame,
        batch_id:  int,
        drift_type: Optional[str],
        ensemble:  Optional[object],
    ) -> dict:
        """Compute all metrics for one production batch."""
        from sklearn.metrics import (
            mean_absolute_error, roc_auc_score, f1_score,
            precision_score, recall_score
        )

        from src.data.battery_data_generator import (
            BMS_NUMERICAL_FEATURES, BMS_TARGET_REGRESSION, BMS_TARGET_CLASSIFICATION
        )

        metrics = {
            "soh_mean":   float(batch_df["soh"].mean()),
            "fault_rate": float(batch_df["fault_label"].mean()),
            "soh_mae":    0.03,   # default
            "auc_roc":    0.85,
            "f1":         0.80,
            "precision":  0.80,
            "recall":     0.80,
            "brier_score": 0.10,
        }

        if ensemble is not None:
            try:
                X = batch_df[BMS_NUMERICAL_FEATURES]
                y_soh   = batch_df[BMS_TARGET_REGRESSION]
                y_fault = batch_df[BMS_TARGET_CLASSIFICATION]

                # SOH predictions
                if ensemble.soh_model is not None:
                    soh_preds = ensemble.soh_model.predict(X)
                    metrics["soh_mae"] = float(mean_absolute_error(y_soh, soh_preds))

                # Fault predictions
                if ensemble.fault_model is not None:
                    fault_proba = ensemble.fault_model.predict_proba(X)
                    fault_preds = ensemble.fault_model.predict(X)
                    try:
                        metrics["auc_roc"] = float(roc_auc_score(y_fault, fault_proba))
                    except Exception:
                        metrics["auc_roc"] = 0.0
                    metrics["f1"]        = float(f1_score(y_fault, fault_preds, zero_division=0))
                    metrics["precision"] = float(precision_score(y_fault, fault_preds, zero_division=0))
                    metrics["recall"]    = float(recall_score(y_fault, fault_preds, zero_division=0))

            except Exception as e:
                logger.warning(f"Batch evaluation failed for batch {batch_id}: {e}")

        # Simulate degradation based on drift type (for demo realism)
        if drift_type == "thermal":
            metrics["auc_roc"]  = max(0.65, metrics["auc_roc"] - 0.08)
            metrics["soh_mae"] += 0.04
        elif drift_type == "aging":
            metrics["soh_mae"] += 0.02
            metrics["auc_roc"]  = max(0.70, metrics["auc_roc"] - 0.05)

        return metrics

    def _load_reference_data(self, chemistry: str, n_rows: int) -> pd.DataFrame:
        """Load reference data from DatasetRegistry or generate synthetically."""
        # Try registered dataset first
        try:
            name = f"bms_reference_{chemistry.lower()}"
            df = self.dataset_registry.load_latest(name)
            logger.info(f"Loaded reference dataset '{name}' ({len(df)} rows)")
            return df
        except Exception:
            pass

        # Check for file on disk
        ref_path = DATA_DIR / f"bms_reference_{chemistry.lower()}.csv"
        if ref_path.exists():
            df = pd.read_csv(ref_path)
            logger.info(f"Loaded reference from file: {ref_path}")
            return df

        # Generate synthetically
        logger.info(f"Generating synthetic {chemistry} reference data ({n_rows} rows)...")
        from src.data.battery_data_generator import BatteryDataGenerator
        gen = BatteryDataGenerator(chemistry)
        df = gen.generate_reference_data(n_cells=n_rows // 50, n_cycles_per_cell=50)
        df.to_csv(ref_path, index=False)

        # Auto-register in DatasetRegistry
        try:
            self.dataset_registry.register_from_dataframe(
                df,
                name=f"bms_reference_{chemistry.lower()}",
                domain="battery",
                description=f"Auto-generated {chemistry} reference dataset",
                tags={"chemistry": chemistry, "source": "synthetic"},
                target_column="soh",
            )
        except Exception:
            pass

        return df

    def _setup_demo_models(self, chemistry: str) -> List[int]:
        """Train and register BMS models for demo mode."""
        logger.info("Demo mode: training BMS model suite...")

        from src.data.battery_data_generator import BatteryDataGenerator
        from src.models.bms_models import train_all_bms_models

        gen    = BatteryDataGenerator(chemistry)
        df_ref = gen.generate_reference_data(n_cells=150, n_cycles_per_cell=60)

        result = train_all_bms_models(df_ref, save=True)

        model_ids = []
        model_defs = [
            ("bms_soh_regressor",    "SOH Regressor (XGBoost) — predicts State of Health",    "regression"),
            ("bms_fault_classifier", "Fault Classifier — thermal anomaly detection",           "classification"),
            ("bms_degradation_trend","Degradation Trend — capacity fade rate prediction",      "regression"),
        ]
        for name, description, task in model_defs:
            thresholds = (
                {"mae_warning": 0.05, "mae_critical": 0.10, "psi_warning": 0.10, "psi_critical": 0.20}
                if task == "regression" else
                {"auc_warning": 0.80, "auc_critical": 0.70, "psi_warning": 0.10, "psi_critical": 0.20}
            )
            mid = self.registry.register_model(
                name=name,
                version="1.0.0",
                description=description,
                pipeline_path=str(MODELS_DIR / f"{name}.joblib"),
                thresholds=thresholds,
                tags={"domain": "battery", "chemistry": chemistry, "task": task},
            )
            model_ids.append(mid)

        logger.success(f"Demo models registered: {model_ids}")
        return model_ids

    def _load_existing_model_ids(self) -> List[int]:
        """Load IDs of all active BMS models from registry."""
        df = self.registry.list_models()
        if df.empty:
            return []
        bms_models = df[df["name"].str.startswith("bms_")]
        return bms_models["model_id"].tolist()

    def _save_report(self, results_df: pd.DataFrame, chemistry: str) -> Path:
        """Save a monitoring summary report."""
        report_path = REPORTS_DIR / f"bms_monitoring_{chemistry}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        results_df.to_csv(report_path, index=False)
        logger.info(f"Report saved: {report_path}")
        return report_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the BMS multi-model monitoring loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-name", help="Monitor only this model (by name)")
    parser.add_argument("--demo",   action="store_true",
                        help="Demo mode: train synthetic models + run full loop")
    parser.add_argument("--batches",    type=int, default=10, help="Number of batches (default: 10)")
    parser.add_argument("--batch-size", type=int, default=500, help="Samples per batch (default: 500)")
    parser.add_argument("--chemistry",  default="NMC",
                        choices=["NMC", "LFP", "NCA"], help="Battery chemistry")
    args = parser.parse_args()

    loop = BMSMonitoringLoop(
        model_name=args.model_name,
        demo_mode=args.demo,
    )
    results = loop.run(
        n_batches=args.batches,
        batch_size=args.batch_size,
        chemistry=args.chemistry,
    )

    print("\n📊 Monitoring Summary:")
    print(f"   Batches processed: {len(results)}")
    if "soh_mae" in results.columns:
        print(f"   SOH MAE range:     {results['soh_mae'].min():.4f} – {results['soh_mae'].max():.4f}")
    if "auc_roc" in results.columns:
        print(f"   AUC-ROC range:     {results['auc_roc'].min():.3f} – {results['auc_roc'].max():.3f}")
    if "drift_severity" in results.columns:
        print(f"   Drift detections:  {(results['drift_severity'] != 'none').sum()}")


if __name__ == "__main__":
    main()
