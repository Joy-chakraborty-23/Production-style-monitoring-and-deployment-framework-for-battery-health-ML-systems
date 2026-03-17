# 🔋 BMS Multi-Model Monitoring System — How to Use

This repository should be used and presented as a **battery/BMS-first ML systems project**.

It contains a reusable monitoring core plus a battery-specific application layer for:
- SoH estimation
- fault detection
- degradation analysis
- dataset and model registration
- edge deployment export paths

> **Placement recommendation:** When demonstrating the repo, always start with the BMS monitoring loop, not the legacy churn example.

---

## 1. Fastest Placement Demo

```bash
cd ml_monitor
python -m src.monitoring.bms_monitoring_loop --demo --batches 10 --chemistry NMC
```

This demo path will:
- set up battery-oriented demo models
- run multiple monitoring batches
- simulate drift conditions
- log monitoring outputs for inspection
- give you a strong battery-first story for presentation

---

## 2. Add Your Own Model

### Interactive
```bash
python -m src.cli.add_model
```

### Scripted
```bash
python -m src.cli.add_model \
  --name my_soh_xgboost \
  --path models/my_model.joblib \
  --task regression \
  --domain battery \
  --version 2.0.0 \
  --description "Custom SOH model trained on LFP cells" \
  --features "voltage,current_a,temperature_c,charge_rate_c,cycle_count"
```

### From Python
```python
from src.monitoring.model_registry import ModelRegistry

registry = ModelRegistry()
model_id = registry.register_model(
    name="my_lfp_soh_model",
    version="1.0.0",
    description="LFP SOH regression model",
    pipeline_path="models/lfp_soh.joblib",
    thresholds={"mae_warning": 0.05, "mae_critical": 0.10},
    tags={"chemistry": "LFP", "team": "bms-team"},
)
print(model_id)
```

---

## 3. Add Your Own Dataset

### Interactive
```bash
python -m src.cli.add_dataset
```

### Scripted
```bash
python -m src.cli.add_dataset \
  --path data/northvolt_lfp_2026.csv \
  --name northvolt_lfp_2026 \
  --domain battery \
  --version 1.0.0 \
  --target soh \
  --description "Northvolt LFP cells 2026 campaign" \
  --tags "chemistry=LFP,source=lab,year=2026"
```

### From Python
```python
from src.data.dataset_registry import DatasetRegistry

registry = DatasetRegistry()
dataset_id = registry.register_from_file(
    path="data/my_battery_data.csv",
    name="my_battery_dataset",
    domain="battery",
    description="Real-world EV battery telemetry",
    tags={"chemistry": "NMC", "source": "field"},
    target_column="soh",
)
print(dataset_id)
```

---

## 4. Battery Chemistry Support

```python
from src.data.battery_data_generator import BatteryDataGenerator

gen_nmc = BatteryDataGenerator("NMC")
ref_nmc = gen_nmc.generate_reference_data(n_cells=300, n_cycles_per_cell=100)

batch = gen_nmc.generate_production_batch(
    n_samples=500,
    batch_id=0,
    drift_type="thermal",
    drift_intensity=0.6,
)
```

Supported chemistry-oriented paths in the repo include examples for:
- NMC
- LFP
- NCA

---

## 5. Ensemble-Style Battery Inference

```python
from src.models.bms_models import EnsembleBMSPredictor
import pandas as pd

ensemble = EnsembleBMSPredictor()
ensemble.load_all_models()

X = pd.read_csv("data/new_battery_readings.csv")
predictions = ensemble.predict_batch(X)
print(predictions.columns)
```

Expected battery-style outputs may include:
- `soh_estimate`
- `soh_uncertainty`
- `fault_probability`
- `fault_flag`
- `capacity_fade_pct`
- `degradation_severity`
- `recommended_action`

---

## 6. Edge Device Export

```python
from src.models.edge_exporter import EdgeExporter
import joblib

exporter = EdgeExporter()
result = exporter.export(
    model=joblib.load("models/bms_soh_regressor.joblib"),
    model_name="soh_regressor",
)
print(result)
```

Use this layer to discuss:
- deployment packaging
- constrained-device compatibility
- ONNX/joblib export choices
- battery analytics near the edge

---

## 7. How to Explain the Repo in Interviews

Use this explanation:

> “The repo started as a reusable monitoring core and evolved into a battery-focused ML monitoring platform. The main value is not only model training, but post-deployment reliability: drift detection, performance tracking, registries, retraining readiness, and deployment hooks.”

Avoid saying:
- “It is mainly a churn project.”
- “It is just a dashboard project.”
- “It is only a single battery model.”

---

## 8. Legacy Example Note

A few generic/churn-oriented modules still exist in the codebase from the earlier reusable-core phase.

Treat them as:
- legacy examples
- reference implementations
- generic monitoring baselines

Do **not** use them as the primary entrypoint in placement discussions.
