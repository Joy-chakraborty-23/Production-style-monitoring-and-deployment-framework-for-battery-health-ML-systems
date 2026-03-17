"""
add_model.py — Interactive CLI for Registering New Models

Allows you to add any trained model to the BMS monitoring system.
Supports: joblib pipelines, XGBoost .json/.ubj files, and custom models.

Usage (interactive):
    python -m src.cli.add_model

Usage (scripted):
    python -m src.cli.add_model \\
        --name my_soh_model \\
        --path models/my_model.joblib \\
        --task regression \\
        --domain battery \\
        --version 2.0.0 \\
        --description "My custom SOH model trained on LFP cells" \\
        --features "voltage,current_a,temperature_c,cycle_count"

After registration, the model is:
  1. Added to the ModelRegistry database
  2. Optionally exported for edge devices
  3. Included in the monitoring loop for drift detection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Supported model formats ────────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".joblib":  "Scikit-learn Pipeline (recommended)",
    ".pkl":     "Pickle file (sklearn, XGBoost, LightGBM, etc.)",
    ".json":    "XGBoost native format",
    ".ubj":     "XGBoost binary format",
    ".onnx":    "ONNX format (ready for edge deployment)",
    ".pt":      "PyTorch model (requires manual wrapper)",
    ".h5":      "Keras/TensorFlow model",
}

SUPPORTED_TASKS    = ["regression", "classification", "anomaly_detection"]
SUPPORTED_DOMAINS  = ["battery", "churn", "ltv", "propensity", "custom"]


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          BMS Monitoring — Add New Model                     ║
║  Registers a new ML model into the monitoring registry      ║
╚══════════════════════════════════════════════════════════════╝
    """)


def interactive_add_model() -> None:
    """Walk the user through adding a model step by step."""
    print_banner()

    # ── Step 1: Model name ────────────────────────────────────────────────
    print("Step 1/8 — Model Name")
    print("  Examples: my_soh_xgboost, lfp_fault_detector, customer_ltv_v2")
    name = input("  Model name: ").strip()
    if not name:
        print("❌ Name cannot be empty.")
        return

    # ── Step 2: Version ───────────────────────────────────────────────────
    print("\nStep 2/8 — Version (default: 1.0.0)")
    version = input("  Version [1.0.0]: ").strip() or "1.0.0"

    # ── Step 3: Model file path ───────────────────────────────────────────
    print("\nStep 3/8 — Model File Path")
    print("  Supported formats:")
    for ext, desc in SUPPORTED_FORMATS.items():
        print(f"    {ext:<10} {desc}")
    model_path_str = input("  Path to model file: ").strip()
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"❌ File not found: {model_path}")
        print("   Tip: Make sure to provide the full or relative path.")
        return

    # ── Step 4: Task type ─────────────────────────────────────────────────
    print("\nStep 4/8 — Task Type")
    for i, t in enumerate(SUPPORTED_TASKS, 1):
        print(f"  {i}. {t}")
    task_input = input("  Select task [1]: ").strip() or "1"
    try:
        task = SUPPORTED_TASKS[int(task_input) - 1]
    except (ValueError, IndexError):
        task = "regression"
    print(f"  → {task}")

    # ── Step 5: Domain ────────────────────────────────────────────────────
    print("\nStep 5/8 — Domain")
    for i, d in enumerate(SUPPORTED_DOMAINS, 1):
        print(f"  {i}. {d}")
    dom_input = input("  Select domain [1=battery]: ").strip() or "1"
    try:
        domain = SUPPORTED_DOMAINS[int(dom_input) - 1]
    except (ValueError, IndexError):
        domain = "battery"
    print(f"  → {domain}")

    # ── Step 6: Feature list ──────────────────────────────────────────────
    print("\nStep 6/8 — Feature Names")
    print("  Enter comma-separated feature column names.")
    print("  BMS defaults: voltage,current_a,temperature_c,charge_rate_c,")
    print("                internal_resistance_mohm,soc,cycle_count")
    feat_input = input("  Features [press Enter for BMS defaults]: ").strip()
    if feat_input:
        features = [f.strip() for f in feat_input.split(",") if f.strip()]
    else:
        features = [
            "voltage", "current_a", "temperature_c", "charge_rate_c",
            "internal_resistance_mohm", "soc", "cycle_count",
        ]
    print(f"  → {len(features)} features")

    # ── Step 7: Description ───────────────────────────────────────────────
    print("\nStep 7/8 — Description")
    description = input("  Short description: ").strip()

    # ── Step 8: Tags ──────────────────────────────────────────────────────
    print("\nStep 8/8 — Tags (optional, key=value pairs)")
    print("  Example: chemistry=NMC,team=bms-team,dataset=northvolt_2026")
    tag_input = input("  Tags [press Enter to skip]: ").strip()
    tags: Dict[str, str] = {}
    if tag_input:
        for pair in tag_input.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                tags[k.strip()] = v.strip()
    tags["domain"] = domain
    tags["task"]   = task

    # ── Preview ───────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("📋 Model Registration Summary")
    print("─" * 60)
    print(f"  Name:        {name}")
    print(f"  Version:     {version}")
    print(f"  File:        {model_path}")
    print(f"  Task:        {task}")
    print(f"  Domain:      {domain}")
    print(f"  Features:    {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
    print(f"  Description: {description or '(none)'}")
    print(f"  Tags:        {tags}")
    print("─" * 60)

    confirm = input("\nRegister this model? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Cancelled.")
        return

    # ── Register ──────────────────────────────────────────────────────────
    _do_register(
        name=name,
        version=version,
        model_path=model_path,
        task=task,
        domain=domain,
        features=features,
        description=description,
        tags=tags,
    )

    # ── Edge export? ──────────────────────────────────────────────────────
    export_edge = input("\nExport for edge device? [y/N]: ").strip().lower()
    if export_edge in ("y", "yes"):
        _do_edge_export(model_path, name, features)


def _do_register(
    name:        str,
    version:     str,
    model_path:  Path,
    task:        str,
    domain:      str,
    features:    List[str],
    description: str,
    tags:        Dict[str, str],
) -> int:
    """Register a model in the ModelRegistry database."""
    try:
        from src.monitoring.model_registry import ModelRegistry
        registry = ModelRegistry()

        thresholds = _get_default_thresholds(task)

        all_tags = {
            **tags,
            "task":          task,
            "domain":        domain,
            "feature_count": str(len(features)),
            "features":      json.dumps(features),
        }

        model_id = registry.register_model(
            name=name,
            version=version,
            description=description or f"{task} model for {domain} domain",
            pipeline_path=str(model_path),
            thresholds=thresholds,
            tags=all_tags,
        )
        print(f"\n✅ Model registered successfully!")
        print(f"   model_id = {model_id}")
        print(f"   Name     = {name} v{version}")
        print(f"   DB path  = models/registry.db")
        print("\nThe model will be included in the next monitoring run.")
        print("To trigger monitoring now: python -m src.monitoring.monitoring_loop --model-id", model_id)
        return model_id

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        print(f"\n❌ Registration failed: {e}")
        print("   Check that the project is set up correctly.")
        return -1


def _do_edge_export(
    model_path:   Path,
    model_name:   str,
    features:     List[str],
) -> None:
    """Export the model for edge devices."""
    print("\nAvailable edge targets:")
    from src.models.edge_exporter import DEVICE_PROFILES
    for i, (key, profile) in enumerate(DEVICE_PROFILES.items(), 1):
        print(f"  {i}. {key:<20} {profile['description']}")

    device_input = input("Select device [1=raspberry_pi_4]: ").strip() or "1"
    try:
        device_key = list(DEVICE_PROFILES.keys())[int(device_input) - 1]
    except (ValueError, IndexError):
        device_key = "raspberry_pi_4"

    try:
        import joblib
        model = joblib.load(model_path)

        from src.models.edge_exporter import EdgeExporter
        exporter = EdgeExporter()
        result   = exporter.export(model, model_name, device_key, feature_names=features)

        print(f"\n✅ Edge export complete!")
        print(f"   Output: {result.output_path}")
        print(f"   Size:   {result.size_kb:.1f} KB")
        print(f"   Format: {result.format}")
        if result.latency_ms > 0:
            print(f"   Latency (this CPU): {result.latency_ms:.2f} ms")
    except Exception as e:
        logger.error(f"Edge export failed: {e}")
        print(f"\n❌ Edge export failed: {e}")


def _get_default_thresholds(task: str) -> Dict[str, float]:
    """Return sensible default thresholds for the given task type."""
    if task == "regression":
        return {
            "mae_warning":  0.05,
            "mae_critical": 0.10,
            "r2_warning":   0.80,
            "r2_critical":  0.60,
            "psi_warning":  0.10,
            "psi_critical": 0.20,
        }
    elif task == "classification":
        return {
            "auc_warning":   0.80,
            "auc_critical":  0.70,
            "f1_warning":    0.75,
            "f1_critical":   0.60,
            "psi_warning":   0.10,
            "psi_critical":  0.20,
        }
    else:
        return {
            "anomaly_rate_warning":  0.05,
            "anomaly_rate_critical": 0.15,
            "psi_warning":  0.10,
            "psi_critical": 0.20,
        }


# ── Scripted (non-interactive) mode ──────────────────────────────────────────

def scripted_add_model(args) -> None:
    """Register a model from CLI arguments (non-interactive)."""
    features = [f.strip() for f in args.features.split(",")] if args.features else [
        "voltage", "current_a", "temperature_c", "charge_rate_c",
        "internal_resistance_mohm", "soc", "cycle_count",
    ]
    tags = {}
    if args.tags:
        for pair in args.tags.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                tags[k.strip()] = v.strip()

    model_path = Path(args.path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    _do_register(
        name=args.name,
        version=args.version,
        model_path=model_path,
        task=args.task,
        domain=args.domain,
        features=features,
        description=args.description or "",
        tags=tags,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add a new model to the BMS Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first use):
  python -m src.cli.add_model

  # Scripted mode:
  python -m src.cli.add_model \\
    --name my_soh_model \\
    --path models/my_model.joblib \\
    --task regression \\
    --domain battery \\
    --version 2.0.0 \\
    --description "Custom SOH model trained on LFP cells" \\
    --features "voltage,current_a,temperature_c,cycle_count"
        """,
    )
    parser.add_argument("--name",        help="Model name (unique identifier)")
    parser.add_argument("--path",        help="Path to model file (.joblib, .pkl, .onnx, etc.)")
    parser.add_argument("--version",     default="1.0.0", help="Semantic version (default: 1.0.0)")
    parser.add_argument("--task",        default="regression",
                        choices=SUPPORTED_TASKS, help="Task type")
    parser.add_argument("--domain",      default="battery",
                        choices=SUPPORTED_DOMAINS, help="Domain")
    parser.add_argument("--description", default="", help="Human-readable description")
    parser.add_argument("--features",    help="Comma-separated feature names")
    parser.add_argument("--tags",        help="Comma-separated key=value tags")

    args = parser.parse_args()

    # If no required args given → interactive mode
    if not args.name or not args.path:
        interactive_add_model()
    else:
        scripted_add_model(args)


if __name__ == "__main__":
    main()
