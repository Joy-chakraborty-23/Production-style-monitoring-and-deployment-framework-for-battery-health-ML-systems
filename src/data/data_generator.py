"""
data_generator.py — Synthetic Telco Churn Dataset with Controllable Drift

Generates realistic customer churn data and simulates 3 types of drift:
  1. Covariate shift   — input feature distributions change
  2. Concept drift     — P(Y|X) changes (model relationship shifts)
  3. Label drift       — prevalence of churn changes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

from src.config import model_config, DATA_DIR


# ── Reproducibility ─────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)


def _sample_categorical(categories: list, probs: list, size: int) -> np.ndarray:
    return RNG.choice(categories, size=size, p=probs)


def generate_reference_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate the clean reference (training) dataset.
    This represents the stable pre-deployment distribution.
    """
    logger.info(f"Generating reference dataset  n={n_samples}")

    df = pd.DataFrame()

    # ── Numerical features ──────────────────────────────────────────────────
    df["tenure"]               = RNG.integers(1, 72, n_samples)
    df["monthly_charges"]      = RNG.normal(65, 30, n_samples).clip(20, 150)
    df["total_charges"]        = df["tenure"] * df["monthly_charges"] * RNG.uniform(0.85, 1.05, n_samples)
    df["num_products"]         = RNG.integers(1, 6, n_samples)
    df["num_support_calls"]    = RNG.poisson(1.5, n_samples).clip(0, 10)
    df["avg_call_duration"]    = RNG.exponential(8, n_samples).clip(1, 60)
    df["days_since_last_login"]= RNG.integers(1, 90, n_samples)
    df["data_usage_gb"]        = RNG.lognormal(2.5, 0.8, n_samples).clip(0.1, 100)
    df["billing_amount_variance"] = RNG.exponential(5, n_samples).clip(0, 50)
    df["customer_lifetime_value"] = (
        df["total_charges"] * RNG.uniform(0.9, 1.1, n_samples)
    ).clip(100, 20000)

    # ── Categorical features ────────────────────────────────────────────────
    df["contract_type"]    = _sample_categorical(
        ["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.20], n_samples
    )
    df["payment_method"]   = _sample_categorical(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        [0.35, 0.22, 0.22, 0.21], n_samples
    )
    df["internet_service"] = _sample_categorical(
        ["DSL", "Fiber optic", "No"], [0.35, 0.44, 0.21], n_samples
    )
    df["tech_support"]     = _sample_categorical(["Yes", "No"], [0.29, 0.71], n_samples)
    df["online_security"]  = _sample_categorical(["Yes", "No"], [0.28, 0.72], n_samples)
    df["paperless_billing"]= _sample_categorical(["Yes", "No"], [0.59, 0.41], n_samples)

    # ── Target (churn) using realistic business logic ───────────────────────
    df["churn"] = _generate_churn_label(df, noise_level=0.05)

    logger.info(f"Reference churn rate: {df['churn'].mean():.3f}")
    return df


def _generate_churn_label(
    df: pd.DataFrame,
    noise_level: float = 0.05,
    contract_multiplier: float = 1.0,
    charge_multiplier: float = 1.0
) -> pd.Series:
    """Deterministic churn logic with configurable multipliers for concept drift.

    Intercept is -3.3 (not -2.5) to produce ~26% churn.
    The original -2.5 caused ~41% churn because all positive feature
    contributions shift the average log-odds well above zero.
    """
    log_odds = (
        -3.3
        + (df["tenure"] < 12).astype(float) * 1.2
        + (df["monthly_charges"] > 80).astype(float) * 0.9 * charge_multiplier
        + (df["num_support_calls"] > 3).astype(float) * 0.8
        + (df["contract_type"] == "Month-to-month").astype(float) * 1.4 * contract_multiplier
        + (df["tech_support"] == "No").astype(float) * 0.5
        + (df["online_security"] == "No").astype(float) * 0.4
        + (df["days_since_last_login"] > 30).astype(float) * 0.6
        + (df["internet_service"] == "Fiber optic").astype(float) * 0.3
        - (df["num_products"] > 3).astype(float) * 0.6
        - df["customer_lifetime_value"] / 15000
    )
    probs = 1 / (1 + np.exp(-log_odds))
    # Add noise
    noise = RNG.uniform(-noise_level, noise_level, len(df))
    probs = (probs + noise).clip(0, 1)
    return (RNG.uniform(size=len(df)) < probs).astype(int)


def generate_production_batch(
    n_samples: int = 500,
    batch_id: int = 0,
    drift_type: Optional[str] = None,
    drift_intensity: float = 0.5,
) -> pd.DataFrame:
    """
    Generate a production batch with optional drift injection.

    Args:
        n_samples:       Records per batch
        batch_id:        Sequential batch number (used for temporal effects)
        drift_type:      None | "covariate" | "concept" | "label" | "sudden" | "gradual"
        drift_intensity: 0.0 (no drift) → 1.0 (severe drift)

    Returns:
        DataFrame with same schema as reference data
    """
    df = pd.DataFrame()

    # ── Intensity scaled by batch for gradual drift ─────────────────────────
    effective_intensity = drift_intensity if drift_type != "gradual" else min(
        drift_intensity * (batch_id / 10), drift_intensity
    )

    # ── Numerical features with optional covariate shift ────────────────────
    charge_shift = 20 * effective_intensity if drift_type in ("covariate", "sudden", "gradual") else 0
    login_shift  = 20 * effective_intensity if drift_type in ("covariate", "sudden", "gradual") else 0

    df["tenure"]               = RNG.integers(1, 72, n_samples)
    df["monthly_charges"]      = (RNG.normal(65 + charge_shift, 30, n_samples)).clip(20, 150)
    df["total_charges"]        = df["tenure"] * df["monthly_charges"] * RNG.uniform(0.85, 1.05, n_samples)
    df["num_products"]         = RNG.integers(1, 6, n_samples)
    df["num_support_calls"]    = RNG.poisson(1.5 + effective_intensity, n_samples).clip(0, 10)
    df["avg_call_duration"]    = RNG.exponential(8, n_samples).clip(1, 60)
    df["days_since_last_login"]= RNG.integers(1, int(90 + login_shift), n_samples)
    df["data_usage_gb"]        = RNG.lognormal(2.5, 0.8, n_samples).clip(0.1, 100)
    df["billing_amount_variance"] = RNG.exponential(5, n_samples).clip(0, 50)
    df["customer_lifetime_value"] = (
        df["total_charges"] * RNG.uniform(0.9, 1.1, n_samples)
    ).clip(100, 20000)

    # ── Categorical features with optional distribution shift ────────────────
    if drift_type in ("covariate", "sudden", "gradual"):
        mtm_prob = min(0.55 + 0.30 * effective_intensity, 0.95)
        remaining = 1 - mtm_prob
        contract_probs = [mtm_prob, remaining * 0.5, remaining * 0.5]
    else:
        contract_probs = [0.55, 0.25, 0.20]

    df["contract_type"]    = _sample_categorical(
        ["Month-to-month", "One year", "Two year"], contract_probs, n_samples
    )
    df["payment_method"]   = _sample_categorical(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        [0.35, 0.22, 0.22, 0.21], n_samples
    )
    df["internet_service"] = _sample_categorical(
        ["DSL", "Fiber optic", "No"], [0.35, 0.44, 0.21], n_samples
    )
    df["tech_support"]     = _sample_categorical(["Yes", "No"], [0.29, 0.71], n_samples)
    df["online_security"]  = _sample_categorical(["Yes", "No"], [0.28, 0.72], n_samples)
    df["paperless_billing"]= _sample_categorical(["Yes", "No"], [0.59, 0.41], n_samples)

    # ── Labels ──────────────────────────────────────────────────────────────
    concept_mult = 1.0 + effective_intensity * 0.8 if drift_type in ("concept", "sudden") else 1.0
    charge_mult  = 1.0 + effective_intensity * 0.5 if drift_type in ("concept",) else 1.0

    if drift_type == "label":
        # Pure label drift: flip some labels independent of features
        df["churn"] = _generate_churn_label(df)
        flip_mask = RNG.uniform(size=n_samples) < (effective_intensity * 0.3)
        df.loc[flip_mask, "churn"] = 1 - df.loc[flip_mask, "churn"]
    else:
        df["churn"] = _generate_churn_label(
            df,
            noise_level=0.05,
            contract_multiplier=concept_mult,
            charge_multiplier=charge_mult
        )

    df["batch_id"]  = batch_id
    df["drift_type"] = drift_type or "none"
    return df


def prepare_train_test_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standard stratified split for model training."""
    from sklearn.model_selection import train_test_split

    feature_cols = model_config.numerical_features + model_config.categorical_features
    X = df[feature_cols].copy()
    y = df[model_config.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_config.test_size,
        stratify=y, random_state=model_config.random_state
    )
    return X_train, X_test, y_train, y_test


def save_datasets(
    reference: pd.DataFrame,
    production_batches: list,
    output_dir: Path = DATA_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output_dir / "reference_data.csv", index=False)
    for i, batch in enumerate(production_batches):
        batch.to_csv(output_dir / f"production_batch_{i:03d}.csv", index=False)
    logger.info(f"Saved reference + {len(production_batches)} production batches to {output_dir}")


if __name__ == "__main__":
    # ── Generate all datasets ─────────────────────────────────────────────
    logger.info("=== Generating full dataset suite ===")

    ref = generate_reference_data(n_samples=5000)

    batches = []
    # Weeks 1–4: clean (no drift)
    for i in range(4):
        batches.append(generate_production_batch(500, batch_id=i, drift_type=None))

    # Week 5–8: gradual covariate drift
    for i in range(4, 8):
        batches.append(generate_production_batch(500, batch_id=i, drift_type="gradual", drift_intensity=0.6))

    # Week 9: sudden concept drift (simulates a market event)
    batches.append(generate_production_batch(500, batch_id=8, drift_type="sudden", drift_intensity=0.9))

    # Weeks 10–12: sustained covariate shift
    for i in range(9, 12):
        batches.append(generate_production_batch(500, batch_id=i, drift_type="covariate", drift_intensity=0.7))

    # Week 13–15: recovery / partial correction
    for i in range(12, 15):
        batches.append(generate_production_batch(500, batch_id=i, drift_type="covariate", drift_intensity=0.2))

    save_datasets(ref, batches)
    logger.success(f"Generated {len(batches)} production batches")
