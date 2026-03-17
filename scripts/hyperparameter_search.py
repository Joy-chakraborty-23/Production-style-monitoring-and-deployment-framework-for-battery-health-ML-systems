"""
scripts/hyperparameter_search.py — Optuna HPO for XGBoost

Searches for optimal XGBoost hyperparameters using Bayesian optimization.
Best params are saved and can be plugged into config.py.

Run:
    python scripts/hyperparameter_search.py --n-trials 50
"""

import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import mlflow
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import model_config, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.data.data_generator import generate_reference_data, prepare_train_test_split
from src.models.train_model import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train, y_train):
    """Optuna objective: maximize cross-validated AUC-ROC."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
    }

    preprocessor = build_preprocessor()
    xgb = XGBClassifier(
        **params,
        random_state=model_config.random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def run_hpo(n_trials: int = 50) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info(f"Starting HPO with {n_trials} trials...")
    ref = generate_reference_data(n_samples=3000)
    X_train, _, y_train, _ = prepare_train_test_split(ref)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best_val = study.best_value
    logger.info(f"Best AUC: {best_val:.4f}")
    logger.info(f"Best params: {best}")

    # Save best params
    out = {"best_auc_cv": round(best_val, 4), "best_params": best}
    out_path = MODELS_DIR / "best_hpo_params.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved best params to {out_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="optuna_hpo_summary"):
        mlflow.log_metric("best_cv_auc", best_val)
        mlflow.log_params(best)
        mlflow.log_artifact(str(out_path))
        mlflow.set_tag("n_trials", n_trials)

    # Print top 5 trials
    df = study.trials_dataframe()
    top5 = df.nlargest(5, "value")[["number", "value", "params_n_estimators",
                                     "params_max_depth", "params_learning_rate"]]
    print("\nTop 5 trials:")
    print(top5.to_string(index=False))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for XGBoost churn model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    result = run_hpo(args.n_trials)
    print(f"\nBest params saved. To apply, update src/config.py with:\n{json.dumps(result['best_params'], indent=2)}")
