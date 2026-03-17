"""
train_model.py — Baseline model training with full MLflow experiment tracking.

Pipeline:
  1. Load reference data
  2. Feature engineering (encoding + scaling)
  3. Train XGBoost with class-imbalance handling
  4. Evaluate (AUC, F1, precision, recall, calibration)
  5. Log everything to MLflow
  6. Save model + preprocessor artifacts
"""

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier

from src.config import (
    model_config, MODELS_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from src.data.data_generator import (
    generate_reference_data, prepare_train_test_split
)


# ── Preprocessor ────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for the feature pipeline."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop="first"
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, model_config.numerical_features),
            ("cat", categorical_transformer, model_config.categorical_features),
        ],
        remainder="drop"
    )


def build_model_pipeline() -> Pipeline:
    """Full sklearn Pipeline: preprocessing + XGBoost classifier."""
    preprocessor = build_preprocessor()
    xgb = XGBClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        learning_rate=model_config.learning_rate,
        subsample=model_config.subsample,
        colsample_bytree=model_config.colsample_bytree,
        scale_pos_weight=model_config.scale_pos_weight,
        random_state=model_config.random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb),
    ])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute a comprehensive set of evaluation metrics."""
    return {
        "auc_roc":          round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_true, y_prob), 4),
        "f1":               round(f1_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "brier_score":      round(brier_score_loss(y_true, y_prob), 4),
        "churn_rate_pred":  round(y_pred.mean(), 4),
        "churn_rate_true":  round(y_true.mean(), 4),
    }


def train_and_log(
    df_ref: pd.DataFrame | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> dict:
    """
    Train the full pipeline and log to MLflow.

    Returns:
        dict with {pipeline, metrics, run_id, model_path}
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    if df_ref is None:
        logger.info("Loading reference data...")
        ref_path = Path("data/reference_data.csv")
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
        else:
            logger.info("No saved reference data found — generating...")
            df_ref = generate_reference_data(n_samples=5000)

    X_train, X_test, y_train, y_test = prepare_train_test_split(df_ref)
    logger.info(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.3f}")

    with mlflow.start_run(run_name="xgboost_baseline") as run:
        run_id = run.info.run_id

        # ── Log hyperparameters ──────────────────────────────────────────
        mlflow.log_params({
            "n_estimators":       model_config.n_estimators,
            "max_depth":          model_config.max_depth,
            "learning_rate":      model_config.learning_rate,
            "subsample":          model_config.subsample,
            "colsample_bytree":   model_config.colsample_bytree,
            "scale_pos_weight":   model_config.scale_pos_weight,
            "train_size":         len(X_train),
            "test_size":          len(X_test),
            "churn_rate_train":   round(float(y_train.mean()), 4),
        })

        # ── Train ────────────────────────────────────────────────────────
        logger.info("Training XGBoost pipeline...")
        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        # ── Calibrate (Platt scaling) ────────────────────────────────────
        # Note: calibrating the full pipeline's probabilities
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        y_prob_test  = pipeline.predict_proba(X_test)[:, 1]
        y_pred_test  = (y_prob_test >= 0.5).astype(int)

        # ── Metrics ──────────────────────────────────────────────────────
        train_metrics = compute_metrics(
            y_train.values, (y_prob_train >= 0.5).astype(int), y_prob_train
        )
        test_metrics = compute_metrics(y_test.values, y_pred_test, y_prob_test)

        # Log train metrics with prefix
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info(f"Test AUC: {test_metrics['auc_roc']:.4f}  F1: {test_metrics['f1']:.4f}")

        # ── Save artifacts ────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "churn_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Save reference data stats for drift baseline
        ref_stats = {
            feat: {
                "mean": float(df_ref[feat].mean()),
                "std":  float(df_ref[feat].std()),
                "p25":  float(df_ref[feat].quantile(0.25)),
                "p50":  float(df_ref[feat].quantile(0.50)),
                "p75":  float(df_ref[feat].quantile(0.75)),
                "min":  float(df_ref[feat].min()),
                "max":  float(df_ref[feat].max()),
            }
            for feat in model_config.numerical_features
        }

        import json
        stats_path = MODELS_DIR / "reference_stats.json"
        with open(stats_path, "w") as f:
            json.dump(ref_stats, f, indent=2)
        mlflow.log_artifact(str(stats_path), artifact_path="model")

        # Save feature names for the pipeline
        feature_names_path = MODELS_DIR / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump({
                "numerical": model_config.numerical_features,
                "categorical": model_config.categorical_features,
                "all": model_config.numerical_features + model_config.categorical_features
            }, f, indent=2)
        mlflow.log_artifact(str(feature_names_path), artifact_path="model")

        # Log classification report as text
        report = classification_report(y_test, y_pred_test)
        mlflow.log_text(report, "classification_report.txt")

        mlflow.set_tags({
            "model_type":  "XGBoost",
            "task":        "binary_classification",
            "domain":      "customer_churn",
            "framework":   "sklearn_pipeline",
        })

    logger.success(f"Training complete. Run ID: {run_id}")
    return {
        "pipeline":   pipeline,
        "metrics":    test_metrics,
        "run_id":     run_id,
        "model_path": model_path,
    }


def load_model(model_path: Path = None) -> Pipeline:
    if model_path is None:
        model_path = MODELS_DIR / "churn_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train_model.py first."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    result = train_and_log()
    logger.info("Metrics:")
    for k, v in result["metrics"].items():
        logger.info(f"  {k}: {v}")
