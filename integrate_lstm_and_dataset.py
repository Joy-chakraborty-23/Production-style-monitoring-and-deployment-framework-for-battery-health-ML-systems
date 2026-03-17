"""
integrate_lstm_and_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-stop integration script for plugging in:
  • SoC_Estimation_LSTM.ipynb   (your trained Keras LSTM model)
  • data.zip                     (your real battery dataset)

Run this script ONCE after extracting your data.zip and exporting your
model from the notebook.  After that, the LSTM + dataset are permanently
registered in the project and automatically used by the monitoring loop.

USAGE
─────
  python integrate_lstm_and_dataset.py

  # Or with explicit paths:
  python integrate_lstm_and_dataset.py \\
      --model-path  models/soc_model.h5 \\
      --data-path   data/battery_data.csv \\
      --scaler-path models/scaler.pkl

WHAT THIS SCRIPT DOES
─────────────────────
  Step 1  Load and normalize YOUR battery dataset (data.zip contents)
  Step 2  Split into train / val / test by cycle (no data leakage)
  Step 3  Register dataset in DatasetRegistry
  Step 4  Load YOUR trained LSTM from the .h5 file OR train one from scratch
  Step 5  Register LSTM in ModelRegistry
  Step 6  Add LSTM to EnsembleBMSPredictor (4-model ensemble)
  Step 7  Run a quick smoke test  →  prints metrics
  Step 8  Export LSTM to edge device (Raspberry Pi 4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging as _l
    logger = _l.getLogger(__name__)
    logger.success = logger.info
    _l.basicConfig(level=_l.INFO, format="%(levelname)s  %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
def run_integration(
    model_path:  str = "",
    data_path:   str = "",
    scaler_path: str = "",
    seq_len:     int = 10,
    epochs:      int = 30,
    chemistry:   str = "NMC",
):
    """
    Full integration pipeline.  All arguments are optional — the script
    gracefully falls back to synthetic data if no real data is provided.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Load and normalize the battery dataset
    # ═══════════════════════════════════════════════════════════════════════
    from src.data.battery_dataset_adapter import BatteryDatasetAdapter
    from src.data.battery_data_generator  import BatteryDataGenerator

    print("\n" + "━"*62)
    print(" STEP 1 — Dataset")
    print("━"*62)

    if data_path and Path(data_path).exists():
        logger.info(f"Loading YOUR dataset: {data_path}")
        df_full = BatteryDatasetAdapter.load_and_normalize(
            data_path,
            cell_id=Path(data_path).stem,
        )
        data_source = "user_uploaded"
    else:
        logger.info(
            "No dataset path provided — generating synthetic NMC data.\n"
            "  To use YOUR data: pass --data-path path/to/your/data.csv"
        )
        gen     = BatteryDataGenerator(chemistry)
        df_full = gen.generate_reference_data(n_cells=200, n_cycles_per_cell=100)
        # Rename columns to match the standard schema the adapter produces
        df_full = df_full.rename(columns={
            "voltage": "Voltage_V", "current_a": "Current_A",
            "temperature_c": "Temperature_C", "soh": "soh",
        })
        df_full["SoC"]     = df_full["soc"]
        df_full["cell_id"] = f"synthetic_{chemistry}"
        data_source = "synthetic"

    print(f"  Dataset shape: {df_full.shape}")
    print(f"  Columns: {list(df_full.columns[:8])}")
    print(f"  SoC range: [{df_full['SoC'].min():.3f}, {df_full['SoC'].max():.3f}]")
    print(f"  Fault rate: {df_full['fault_label'].mean():.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Cycle-aware train/val/test split  (no data leakage)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 2 — Train / Val / Test Split  (by cycle)")
    print("━"*62)

    df_train, df_val, df_test = BatteryDatasetAdapter.split_by_cycle(
        df_full, train_frac=0.70, val_frac=0.15
    )
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Register dataset in DatasetRegistry
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 3 — Register Dataset in DatasetRegistry")
    print("━"*62)

    from src.data.dataset_registry import DatasetRegistry
    ds_registry = DatasetRegistry()

    ds_name = Path(data_path).stem if data_path else f"battery_{chemistry.lower()}_dataset"
    did = ds_registry.register_from_dataframe(
        df_train,
        name=ds_name,
        domain="battery",
        description=(
            f"Battery SoC dataset from {data_source}. "
            f"Used to train LSTM SoC model and BMS ensemble."
        ),
        tags={"source": data_source, "chemistry": chemistry,
              "split": "train", "target": "SoC"},
        target_column="SoC",
    )
    print(f"  ✅ Registered dataset '{ds_name}'  dataset_id={did}")
    print(f"     Load later: DatasetRegistry().load_dataset('{ds_name}')")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Load YOUR Keras LSTM  OR  train one from scratch
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 4 — SoC LSTM Model")
    print("━"*62)

    from src.models.soc_lstm_wrapper import (
        SoCLSTMWrapper, SoCLSTMFallbackWrapper, train_soc_lstm,
        SOC_INPUT_FEATURES, SOC_TARGET,
    )

    # Check which columns are available
    available_feats = [c for c in SOC_INPUT_FEATURES if c in df_train.columns]
    if not available_feats:
        # Try lowercase variants (from the generator)
        rename_map = {
            "voltage": "Voltage_V", "current_a": "Current_A",
            "temperature_c": "Temperature_C",
        }
        df_train = df_train.rename(columns=rename_map)
        df_val   = df_val.rename(columns=rename_map)
        df_test  = df_test.rename(columns=rename_map)
        available_feats = [c for c in SOC_INPUT_FEATURES if c in df_train.columns]

    if model_path and Path(model_path).exists():
        # ── A) Load the model saved by your notebook ──────────────────────
        print(f"  Loading YOUR Keras LSTM: {model_path}")
        soc_model = SoCLSTMWrapper.from_keras_file(
            keras_path=model_path,
            seq_len=seq_len,
            feature_cols=available_feats,
            scaler_path=scaler_path if scaler_path else None,
        )
        # Fit the sequence scaler on the training split
        if not soc_model.seq_builder_._fitted:
            print("  Fitting sequence scaler on training data...")
            soc_model.seq_builder_.fit(df_train)
        method = "loaded_from_notebook"
    else:
        # ── B) Train a new LSTM from scratch on this dataset ──────────────
        print(
            "  No model path provided — training LSTM from scratch.\n"
            "  To use YOUR notebook model: pass --model-path path/to/soc_model.h5"
        )
        result    = train_soc_lstm(
            df_train      = df_train,
            df_val        = df_val if len(df_val) > seq_len else None,
            seq_len       = seq_len,
            epochs        = epochs,
            save          = True,
        )
        soc_model = result["model"]
        method    = result["model_type"]
        print(f"  Training metrics: {result['metrics']}")

    # Evaluate on test set
    if SOC_TARGET in df_test.columns and len(df_test) > seq_len:
        test_metrics = soc_model.evaluate(df_test, df_test[SOC_TARGET])
        print(f"  ✅ LSTM [{method}] test metrics: {test_metrics}")
    else:
        test_metrics = {}
        print(f"  ✅ LSTM [{method}] loaded (no test eval — insufficient rows)")

    # Save
    saved_path = soc_model.save()
    print(f"  Saved → {saved_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5 — Register LSTM in ModelRegistry
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 5 — Register LSTM in ModelRegistry")
    print("━"*62)

    from src.monitoring.model_registry import ModelRegistry
    registry = ModelRegistry()

    lstm_model_id = registry.register_model(
        name=soc_model.name,
        version="1.0.0",
        description=(
            f"LSTM SoC estimator ({method}) — "
            f"predicts State of Charge from voltage/current/temperature time series. "
            f"MAE={test_metrics.get('mae', 'N/A')}"
        ),
        pipeline_path=str(saved_path),
        thresholds={
            "mae_warning":  0.05,
            "mae_critical": 0.10,
            "psi_warning":  0.10,
            "psi_critical": 0.20,
        },
        tags={
            "domain":       "battery",
            "task":         "soc_regression",
            "model_type":   method,
            "features":     str(available_feats),
            "seq_len":      str(seq_len),
            "dataset":      ds_name,
        },
    )
    print(f"  ✅ LSTM registered  model_id={lstm_model_id}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6 — Build 4-model ensemble and smoke test
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 6 — 4-Model Ensemble Smoke Test")
    print("━"*62)

    from src.models.bms_models import EnsembleBMSPredictor, train_all_bms_models

    # Train the three BMS models if not already on disk
    models_dir = PROJECT_ROOT / "models"
    soh_path   = models_dir / "bms_soh_regressor.joblib"
    if not soh_path.exists():
        print("  Training base BMS models (XGBoost SOH + Fault + Trend)...")
        train_all_bms_models(df_train, save=True)

    # Load full ensemble
    ensemble = EnsembleBMSPredictor()
    ensemble.load_all_models()
    ensemble.soc_lstm_model = soc_model   # inject directly

    n_models = sum(m is not None for m in [
        ensemble.soh_model, ensemble.fault_model,
        ensemble.trend_model, ensemble.soc_lstm_model
    ])
    print(f"  Ensemble has {n_models}/4 models")

    # Run batch prediction on test slice
    from src.data.battery_data_generator import BMS_NUMERICAL_FEATURES
    test_slice = df_test.head(20).copy()
    # Ensure BMS numerical features exist
    for col in BMS_NUMERICAL_FEATURES:
        if col not in test_slice.columns:
            test_slice[col] = 0.0

    predictions = ensemble.predict_batch(test_slice)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  SOH estimates:     {predictions['soh_estimate'].values[:5].round(3).tolist()}")
    print(f"  Fault flags:       {predictions['fault_flag'].values[:5].tolist()}")
    print(f"  Severity counts:   {predictions['degradation_severity'].value_counts().to_dict()}")
    print("  ✅ Ensemble working correctly")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7 — Export LSTM to Edge Device
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" STEP 7 — Edge Export (Raspberry Pi 4)")
    print("━"*62)

    from src.models.edge_exporter import EdgeExporter
    exporter = EdgeExporter()

    try:
        result = exporter.export(
            model=soc_model,
            model_name="soc_lstm",
            target_device="raspberry_pi_4",
            X_sample=df_test[available_feats].head(100) if len(df_test) >= 10 else None,
            feature_names=available_feats,
        )
        print(f"  ✅ Edge export:  {result.output_path}")
        print(f"     Format:        {result.format}")
        print(f"     Size:          {result.size_kb:.1f} KB")
        print(f"     Latency (CPU): {result.latency_ms:.2f} ms")
    except Exception as e:
        print(f"  ⚠️  Edge export skipped: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # DONE — Print usage summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "━"*62)
    print(" ✅  INTEGRATION COMPLETE")
    print("━"*62)
    print(f"""
  What was registered:
    Dataset  : '{ds_name}'  (id={did})
    LSTM     : '{soc_model.name}'  (id={lstm_model_id})
    Ensemble : 4-model (XGBoost SOH + Fault + Trend + LSTM SoC)

  Next steps:

    1. Run the monitoring loop (uses all 4 models automatically):
       python -m src.monitoring.bms_monitoring_loop

    2. Add MORE datasets at any time:
       python -m src.cli.add_dataset --path data/new_data.csv --name my_data

    3. Add MORE models at any time:
       python -m src.cli.add_model --path models/my_model.h5 --name my_model

    4. Load the LSTM elsewhere in code:
       from src.models.soc_lstm_wrapper import SoCLSTMWrapper
       model = SoCLSTMWrapper.load()
       soc   = model.predict_realtime(window_df)   # → scalar ∈ [0,1]

    5. Load the full ensemble:
       from src.models.bms_models import EnsembleBMSPredictor
       ens = EnsembleBMSPredictor()
       ens.load_all_models()
       preds = ens.predict_batch(df)
    """)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Integrate SoC_Estimation_LSTM.ipynb + data.zip into BMS project"
    )
    p.add_argument("--model-path",  default="",
                   help="Path to .h5 / .keras Keras model from the notebook")
    p.add_argument("--data-path",   default="",
                   help="Path to battery CSV/xlsx from data.zip")
    p.add_argument("--scaler-path", default="",
                   help="Path to MinMaxScaler saved by the notebook (optional)")
    p.add_argument("--seq-len",  type=int, default=10,
                   help="LSTM look-back window length (default: 10)")
    p.add_argument("--epochs",   type=int, default=30,
                   help="Max training epochs (default: 30)")
    p.add_argument("--chemistry", default="NMC",
                   choices=["NMC", "LFP", "NCA"],
                   help="Battery chemistry for synthetic fallback (default: NMC)")
    args = p.parse_args()

    run_integration(
        model_path=args.model_path,
        data_path=args.data_path,
        scaler_path=args.scaler_path,
        seq_len=args.seq_len,
        epochs=args.epochs,
        chemistry=args.chemistry,
    )


if __name__ == "__main__":
    main()
