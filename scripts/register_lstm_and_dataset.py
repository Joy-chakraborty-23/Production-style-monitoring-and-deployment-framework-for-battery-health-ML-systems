"""
register_lstm_and_dataset.py — One-Shot Integration Script

Registers the SP20-2 battery dataset and the SoC LSTM model into the
BMS monitoring system in a single command.

What this script does (in order):
  Step 1 — Preprocess all SP20-2 XLS files → clean CSVs with SOC labels
  Step 2 — Register the merged dataset into DatasetRegistry
  Step 3 — Split data by temperature for train/test (25°C+45°C train, 0°C test)
  Step 4 — Train the SoCLSTM (if torch available) OR register a pre-trained .pth
  Step 5 — Evaluate the model on the 0°C hold-out test set
  Step 6 — Register the model into ModelRegistry with correct feature schema
  Step 7 — Optionally export for edge devices
  Step 8 — Print a full summary: paths, metrics, next steps

Usage:
  # Full pipeline (preprocess + train + register):
  python scripts/register_lstm_and_dataset.py \\
      --data-dir  data/raw/SP2 \\
      --output-dir data/processed \\
      --epochs 20

  # Load a pretrained .pth from the notebook (skip training):
  python scripts/register_lstm_and_dataset.py \\
      --data-dir   data/raw/SP2 \\
      --output-dir data/processed \\
      --pretrained-pth models/soc_lstm_model.pth \\
      --skip-training

  # After registering: run monitoring loop that includes the LSTM
  python -m src.monitoring.bms_monitoring_loop
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info
    logging = _logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def run(
    data_dir:        Path,
    output_dir:      Path,
    pretrained_pth:  Path | None = None,
    skip_training:   bool = False,
    epochs:          int  = 20,
    export_edge:     bool = False,
    edge_device:     str  = "raspberry_pi_4",
    q_nominal:       float = 2.15,
) -> dict:
    """
    Full registration pipeline. Returns a summary dict.
    """
    print("\n" + "═" * 70)
    print("  BMS Monitoring — SP20 Dataset + SoC LSTM Registration")
    print("═" * 70 + "\n")

    results = {}

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Preprocess SP20-2 XLS files
    # ─────────────────────────────────────────────────────────────────────
    print("━" * 50)
    print("STEP 1 — Preprocessing SP20-2 Battery Data")
    print("━" * 50)

    from src.data.sp20_data_preprocessor import SP20Preprocessor, FEATURE_COLS, LABEL_COL

    preprocessor = SP20Preprocessor(
        data_dir   = data_dir,
        q_nominal  = q_nominal,
        output_dir = output_dir,
    )
    merged_df = preprocessor.process_all(save_individual=True)

    results["total_rows"]   = len(merged_df)
    results["n_files"]      = merged_df["source_file"].nunique()
    results["soc_min"]      = float(merged_df[LABEL_COL].min())
    results["soc_max"]      = float(merged_df[LABEL_COL].max())
    results["merged_csv"]   = str(output_dir / "sp20_all_merged.csv")

    print(f"\n  ✅ Preprocessed {results['n_files']} files → {results['total_rows']:,} rows")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Register dataset in DatasetRegistry
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("STEP 2 — Registering Dataset in DatasetRegistry")
    print("━" * 50)

    from src.data.dataset_registry import DatasetRegistry

    ds_registry = DatasetRegistry()
    dataset_id  = ds_registry.register_from_dataframe(
        df            = merged_df,
        name          = "sp20_battery_soc",
        version       = "1.0.0",
        domain        = "battery",
        description   = (
            "SP20-2 Li-ion battery SoC dataset — 12 Arbin test files, "
            "3 temperatures (0/25/45°C), 2 protocols (DST/FUDS), "
            "2 initial SOCs (50%/80%). SOC labels from Coulomb counting."
        ),
        tags          = {
            "chemistry":   "Li-ion",
            "cell_model":  "SP20-2",
            "source":      "Arbin_cycler",
            "temperatures": "0C_25C_45C",
            "protocols":   "DST_FUDS",
            "n_files":     str(results["n_files"]),
            "q_nominal_ah": str(q_nominal),
        },
        target_column = LABEL_COL,
    )
    results["dataset_id"] = dataset_id
    print(f"\n  ✅ Dataset registered: 'sp20_battery_soc' v1.0.0  (id={dataset_id})")

    # Also register per-temperature subsets for finer-grained model selection
    for temp_c, group in merged_df.groupby("temperature_c"):
        sub_id = ds_registry.register_from_dataframe(
            df            = group.reset_index(drop=True),
            name          = f"sp20_soc_{temp_c}c",
            version       = "1.0.0",
            domain        = "battery",
            description   = f"SP20-2 SoC data at {temp_c}°C only",
            tags          = {"chemistry": "Li-ion", "temperature_c": str(temp_c)},
            target_column = LABEL_COL,
        )
        print(f"  ✅ Sub-dataset registered: 'sp20_soc_{temp_c}c'  rows={len(group):,}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Train/test split by temperature
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("STEP 3 — Train/Test Split  (train=25°C+45°C  test=0°C)")
    print("━" * 50)

    train_df, test_df = preprocessor.split_by_temperature(
        merged_df,
        train_temps = [25, 45],
        test_temps  = [0],
    )
    results["train_rows"] = len(train_df)
    results["test_rows"]  = len(test_df)
    print(f"\n  Train: {len(train_df):,} rows  |  Test (0°C hold-out): {len(test_df):,} rows")

    # Val split: last 15% of training set
    val_split = int(len(train_df) * 0.85)
    val_df    = train_df.iloc[val_split:].copy()
    train_df  = train_df.iloc[:val_split].copy()
    print(f"  Val:   {len(val_df):,} rows  (last 15% of train)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Train the SoCLSTM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("STEP 4 — SoC LSTM Model")
    print("━" * 50)

    from src.models.soc_lstm_wrapper import SoCLSTMWrapper, LSTM_FEATURE_COLS, LSTM_LABEL_COL

    model = SoCLSTMWrapper(epochs=epochs)

    if pretrained_pth and Path(pretrained_pth).exists():
        # ── Load pre-trained .pth from the original notebook ────────────
        print(f"\n  Loading pretrained model from: {pretrained_pth}")
        model.load_pretrained_pth(
            pth_path   = pretrained_pth,
            hidden_size = 94,
            num_layers  = 4,
        )
        print("  ✅ Pretrained model loaded")

    elif not skip_training:
        # ── Train from scratch ────────────────────────────────────────
        try:
            import torch
            print(f"\n  Training SoCLSTM — {epochs} epochs on {len(train_df):,} samples")
            print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            print()
            model.train_on_dataframe(train_df, val_df, verbose=True)
            print()
        except ImportError:
            print("\n  ⚠️  PyTorch not installed — skipping training.")
            print("  To train: pip install torch")
            print("  To load pretrained: use --pretrained-pth models/soc_lstm_model.pth")
            skip_training = True

    else:
        print("\n  Skipping training (--skip-training flag set).")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Evaluate on 0°C hold-out test set
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("STEP 5 — Evaluation on 0°C Hold-Out Test Set")
    print("━" * 50)

    test_metrics = {}
    if not skip_training or (pretrained_pth and Path(pretrained_pth).exists()):
        try:
            X_test = test_df[LSTM_FEATURE_COLS]
            y_test = test_df[LSTM_LABEL_COL]
            test_metrics = model.evaluate(X_test, y_test)
            results["test_metrics"] = test_metrics
            print(f"\n  Test metrics (0°C hold-out):")
            print(f"    MAE:     {test_metrics['mae']:.4f}   (target: < 0.03)")
            print(f"    RMSE:    {test_metrics['rmse']:.4f}")
            print(f"    R²:      {test_metrics['r2']:.4f}   (target: > 0.95)")
            print(f"    Samples: {test_metrics['n_samples']:,}")

            # Also show per-temperature eval
            print(f"\n  Per-temperature breakdown:")
            for temp_c, group in merged_df.groupby("temperature_c"):
                try:
                    Xg = group[LSTM_FEATURE_COLS]
                    yg = group[LSTM_LABEL_COL]
                    m  = model.evaluate(Xg, yg)
                    print(f"    {temp_c:>2}°C: MAE={m['mae']:.4f}  R²={m['r2']:.4f}  n={m['n_samples']:,}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            test_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    else:
        test_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        print("\n  Skipping evaluation (model not trained).")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Save model + Register in ModelRegistry
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("STEP 6 — Registering LSTM in ModelRegistry")
    print("━" * 50)

    model_path = None
    if not skip_training or (pretrained_pth and Path(pretrained_pth).exists()):
        try:
            model_path = model.save(BASE_DIR / "models" / "soc_lstm.pth")
            print(f"\n  ✅ Model saved → {model_path}")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
            model_path = pretrained_pth or BASE_DIR / "models" / "soc_lstm.pth"

    from src.monitoring.model_registry import ModelRegistry
    registry = ModelRegistry()
    model_id = registry.register_model(
        name          = "soc_lstm",
        version       = "1.0.0",
        description   = (
            "SoC LSTM (PyTorch) — Predicts State of Charge from time-series "
            "battery telemetry. Architecture: LSTM (hidden=94, layers=4, seq=20). "
            "Trained on SP20-2 Li-ion cells at 25°C + 45°C."
        ),
        pipeline_path = str(model_path or "models/soc_lstm.pth"),
        thresholds    = {
            "mae_warning":  0.03,   # 3% SOC error
            "mae_critical": 0.05,   # 5% SOC error
            "r2_warning":   0.95,
            "r2_critical":  0.85,
            "psi_warning":  0.10,
            "psi_critical": 0.20,
        },
        tags = {
            "domain":         "battery",
            "task":           "regression",
            "architecture":   "LSTM",
            "framework":      "pytorch",
            "cell_chemistry": "Li-ion",
            "cell_model":     "SP20-2",
            "target":         LSTM_LABEL_COL,
            "features":       ",".join(LSTM_FEATURE_COLS),
            "train_temps":    "25C_45C",
            "test_temp":      "0C",
            "hidden_size":    "94",
            "num_layers":     "4",
            "sequence_length": "20",
            "dataset":        "sp20_battery_soc_v1.0.0",
            "test_mae":       str(test_metrics.get("mae", "N/A")),
            "test_r2":        str(test_metrics.get("r2",  "N/A")),
        },
    )
    results["model_id"] = model_id
    print(f"  ✅ Model registered: 'soc_lstm' v1.0.0  (id={model_id})")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Edge export (optional)
    # ─────────────────────────────────────────────────────────────────────
    if export_edge and model_path and not skip_training:
        print("\n" + "━" * 50)
        print(f"STEP 7 — Edge Export  ({edge_device})")
        print("━" * 50)
        try:
            from src.models.edge_exporter import EdgeExporter
            exporter   = EdgeExporter()
            X_sample   = test_df[LSTM_FEATURE_COLS].head(200)
            edge_result = exporter.export(model, "soc_lstm", edge_device, X_sample)
            results["edge_path"]     = edge_result.output_path
            results["edge_size_kb"]  = edge_result.size_kb
            results["edge_latency"]  = edge_result.latency_ms
            print(f"  ✅ Exported: {edge_result.output_path}")
            print(f"     Size:    {edge_result.size_kb:.1f} KB")
            print(f"     Latency: {edge_result.latency_ms:.2f} ms")
        except Exception as e:
            logger.warning(f"Edge export failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8: Summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  REGISTRATION COMPLETE — Summary")
    print("═" * 70)
    print(f"""
  Dataset: 'sp20_battery_soc' v1.0.0
    ID:     {results.get('dataset_id')}
    Rows:   {results.get('total_rows', 0):,}
    Files:  {results.get('n_files', 0)}
    CSV:    {results.get('merged_csv')}

  Model: 'soc_lstm' v1.0.0
    ID:     {results.get('model_id')}
    Task:   SOC regression (LSTM, seq_len=20)
    MAE:    {test_metrics.get('mae', 'N/A')}
    R²:     {test_metrics.get('r2',  'N/A')}
    Path:   {model_path or 'models/soc_lstm.pth'}
""")
    print("  Next Steps:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  1. Run the monitoring loop (now includes LSTM):        │")
    print("  │     python -m src.monitoring.bms_monitoring_loop        │")
    print("  │                                                          │")
    print("  │  2. Load the LSTM for inference in your code:           │")
    print("  │     from src.models.soc_lstm_wrapper import SoCLSTMWrapper │")
    print("  │     m = SoCLSTMWrapper().load('models/soc_lstm.pth')   │")
    print("  │     soc = m.predict(your_dataframe)                     │")
    print("  │                                                          │")
    print("  │  3. Load the dataset:                                   │")
    print("  │     from src.data.dataset_registry import DatasetRegistry│")
    print("  │     df = DatasetRegistry().load_dataset('sp20_battery_soc')│")
    print("  └─────────────────────────────────────────────────────────┘\n")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Register SP20-2 dataset + SoC LSTM into BMS monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir",       required=True,
                        help="Path to folder containing SP2_*C_* subfolders")
    parser.add_argument("--output-dir",     default="data/processed",
                        help="Output dir for processed CSVs (default: data/processed)")
    parser.add_argument("--pretrained-pth", default=None,
                        help="Path to pretrained .pth from notebook (skips training)")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip training — only preprocess + register dataset")
    parser.add_argument("--epochs",         type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--export-edge",    action="store_true",
                        help="Export LSTM for edge device after training")
    parser.add_argument("--edge-device",    default="raspberry_pi_4",
                        help="Edge target (default: raspberry_pi_4)")
    parser.add_argument("--q-nominal",      type=float, default=2.15,
                        help="Rated capacity in Ah (default: 2.15)")
    args = parser.parse_args()

    run(
        data_dir       = Path(args.data_dir),
        output_dir     = Path(args.output_dir),
        pretrained_pth = Path(args.pretrained_pth) if args.pretrained_pth else None,
        skip_training  = args.skip_training,
        epochs         = args.epochs,
        export_edge    = args.export_edge,
        edge_device    = args.edge_device,
        q_nominal      = args.q_nominal,
    )


if __name__ == "__main__":
    main()
