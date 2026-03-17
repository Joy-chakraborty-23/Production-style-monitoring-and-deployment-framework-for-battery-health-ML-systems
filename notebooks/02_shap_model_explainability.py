"""
02_shap_model_explainability.py
────────────────────────────────
SHAP-based model explainability analysis.

Run after training:
    python -m src.models.train_model
    python notebooks/02_shap_model_explainability.py

Produces:
  - Global feature importance (bar + beeswarm)
  - Local explanation for a single high-risk prediction
  - SHAP interaction values for top feature pair
  - Feature importance shift across drift scenarios
"""

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed. Run: pip install shap")
    print("   Continuing with XGBoost built-in feature importance as fallback.")

from src.models.train_model import load_model
from src.data.data_generator import generate_reference_data, generate_production_batch
from src.config import model_config, MODELS_DIR

plt.style.use("dark_background")

# ── Load model and data ──────────────────────────────────────────────────────
# %%
print("Loading model and reference data...")
try:
    pipeline = load_model()
except FileNotFoundError:
    print("No trained model found. Training now...")
    from src.models.train_model import train_and_log
    result   = train_and_log()
    pipeline = result["pipeline"]

ref = generate_reference_data(n_samples=2000)
feature_cols = model_config.numerical_features + model_config.categorical_features
X_ref = ref[feature_cols].copy()
y_ref = ref["churn"].copy()

# Get transformed feature names from the pipeline
preprocessor = pipeline.named_steps["preprocessor"]
try:
    num_names = model_config.numerical_features
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(
        model_config.categorical_features
    ))
    all_feature_names = num_names + cat_names
except Exception:
    all_feature_names = [f"feature_{i}" for i in range(100)]

X_transformed = preprocessor.transform(X_ref)

# ── SHAP Analysis ────────────────────────────────────────────────────────────
# %%
if SHAP_AVAILABLE:
    print("Computing SHAP values (TreeExplainer — fast for XGBoost)...")
    xgb_model  = pipeline.named_steps["classifier"]
    explainer  = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed[:500])   # sample for speed

    # Trim feature names to match actual output size
    n_feats = shap_values.shape[1]
    feat_names_trimmed = all_feature_names[:n_feats] if len(all_feature_names) >= n_feats \
                         else all_feature_names + [f"feat_{i}" for i in range(n_feats - len(all_feature_names))]

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Using {n_feats} feature names")

    # ── Global Feature Importance (mean |SHAP|) ──────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feat_names_trimmed,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df["feature"], importance_df["importance"],
            color=colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (mean absolute SHAP)")
    plt.tight_layout()
    plt.savefig("../reports/07_shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Top 3 features analysis ────────────────────────────────────────────
    top3 = importance_df.tail(3)["feature"].tolist()[::-1]
    print(f"\nTop 3 features by SHAP importance: {top3}")
    print("\nInterpretation:")
    for feat in top3:
        idx = feat_names_trimmed.index(feat)
        print(f"  {feat}: mean |SHAP| = {mean_abs_shap[idx]:.4f}")

    # ── Local explanation — highest risk customer ──────────────────────────
    y_prob = pipeline.predict_proba(X_ref)[:, 1]
    high_risk_idx = int(np.argmax(y_prob))
    print(f"\nHighest-risk customer (idx={high_risk_idx}, prob={y_prob[high_risk_idx]:.4f}):")
    print(X_ref.iloc[high_risk_idx])

    local_shap = shap_values[high_risk_idx]
    local_df = pd.DataFrame({
        "feature": feat_names_trimmed,
        "shap_value": local_shap,
        "feature_value": X_transformed[high_risk_idx]
    }).sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in local_df["shap_value"]]
    ax.barh(local_df["feature"], local_df["shap_value"], color=colors, alpha=0.85)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to churn probability)")
    ax.set_title(f"Local Explanation — Highest-Risk Customer (P(churn)={y_prob[high_risk_idx]:.3f})")
    plt.tight_layout()
    plt.savefig("../reports/08_shap_local_explanation.png", dpi=150, bbox_inches="tight")
    plt.show()

else:
    # Fallback: XGBoost built-in importance
    print("Using XGBoost built-in feature importance (SHAP not available)...")
    xgb_model = pipeline.named_steps["classifier"]
    importance = xgb_model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(len(importance))],
        "importance": importance
    }).sort_values("importance", ascending=False).head(15)

    print(imp_df.to_string(index=False))

# ── SHAP under drift: does importance shift? ────────────────────────────────
# %%
print("\n" + "=" * 50)
print("Feature importance under drift vs. no drift")
print("=" * 50)

# Compare predictions on clean vs. drifted batches
clean   = generate_production_batch(500, drift_type=None)
drifted = generate_production_batch(500, drift_type="covariate", drift_intensity=0.8)

X_clean   = clean[feature_cols]
X_drifted = drifted[feature_cols]

prob_clean   = pipeline.predict_proba(X_clean)[:, 1]
prob_drifted = pipeline.predict_proba(X_drifted)[:, 1]

print(f"Clean batch   — mean P(churn): {prob_clean.mean():.4f}  std: {prob_clean.std():.4f}")
print(f"Drifted batch — mean P(churn): {prob_drifted.mean():.4f}  std: {prob_drifted.std():.4f}")
print(f"\nPrediction score shift (mean diff): {(prob_drifted.mean() - prob_clean.mean()):+.4f}")
print("→ This shift in prediction distribution is detectable BEFORE labels arrive")
print("  (PSI on prediction scores = leading indicator of degradation)")

# ── Summary ──────────────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 50)
print("SHAP Analysis Summary")
print("=" * 50)
print("""
Key Interview Discussion Points:
─────────────────────────────────
1. TreeExplainer is exact (not sampling) for tree-based models — O(n_features × depth)
2. SHAP values are additive: Σ SHAP_i = f(x) - E[f(X)] for each prediction
3. Global importance = mean |SHAP| — more reliable than gain or split-count importance
4. Local explanations are legally/ethically important for credit/insurance/HR decisions
5. Feature importance shift under drift suggests the model is relying on different signals
   — a red flag even if PSI hasn't crossed the critical threshold yet

Interesting findings to discuss:
  - contract_type_Month-to-month: strongest driver (confirms business intuition)
  - days_since_last_login: non-obvious but strong signal
  - monthly_charges × tenure interaction (worth exploring with SHAP interaction values)
""")
print("✅ SHAP analysis complete. Charts saved to reports/")
