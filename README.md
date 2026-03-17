# 🔋 Battery ML Monitoring Platform for BMS

> **Production-style monitoring and deployment framework for battery-health ML systems** — supporting battery dataset onboarding, multi-model inference, drift detection, performance tracking, retraining workflows, registry support, and edge deployment hooks.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Why This Project Exists](#why-this-project-exists)
- [What the Platform Does](#what-the-platform-does)
- [Architecture](#architecture)
- [Battery-Specific Capabilities](#battery-specific-capabilities)
- [Quick Start](#quick-start)
- [Recommended Placement Demo](#recommended-placement-demo)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Legacy Example Notice](#legacy-example-notice)
- [Interview Positioning](#interview-positioning)
- [Resume Bullets](#resume-bullets)

---

## Project Overview

This repository is a **BMS-first ML systems project**.

It was originally built as a reusable ML monitoring core for tabular models, then extended into a **battery-focused monitoring platform** for Battery Management System (BMS) use cases such as:
- **State of Health (SoH) estimation**
- **Fault detection**
- **Degradation monitoring**
- **Multi-model battery intelligence workflows**
- **Deployment-oriented export for edge environments**

The result is a project that is not just about training a model, but about **what happens after deployment**: detecting drift, tracking model quality, managing datasets and models, triggering retraining, and preparing models for practical deployment.

---

## Why This Project Exists

Battery ML systems operate in environments where data distributions can shift because of:
- temperature variation
- chemistry differences (NMC / LFP / NCA)
- aging and cycle count
- charging patterns
- sensor noise and field conditions

A model that performs well in training can quietly become unreliable once deployed. This project addresses that post-training gap by providing a **production-style observability and monitoring layer for battery ML systems**.

---

## What the Platform Does

| Layer | Capability |
|------|------|
| **Data Layer** | Registers datasets, adapts battery schemas, preprocesses SP20 battery data |
| **Model Layer** | Supports battery models for SoH, fault detection, degradation analysis, and LSTM integration |
| **Monitoring Layer** | Detects drift, tracks performance, computes BMS-oriented impact, logs fleet health |
| **Ops Layer** | Supports registries, retraining workflows, test coverage, Docker setup, CLI tools |
| **Deployment Layer** | Provides edge export hooks for Raspberry Pi / MCU / mobile-oriented deployment paths |

---

## Architecture

```text
Battery Dataset / Telemetry
        │
        ▼
Dataset Registry ──► Battery Dataset Adapter / SP20 Preprocessor
        │
        ▼
Battery Models (SoH / Fault / Trend / LSTM)
        │
        ▼
Inference + Monitoring Layer
   ├── Drift Detection
   ├── Performance Monitoring
   ├── BMS ROI / Risk Estimation
   ├── Alerting
   └── Retraining Triggers
        │
        ▼
Model Registry / Fleet Health Export / Reports
        │
        ▼
Dashboard / API / Edge Export
```

### Design philosophy
This repository is best understood as:
1. a **reusable ML monitoring core**, and
2. a **battery-specific application layer** built on top of that core.

That is why some generic tabular monitoring components still exist in the codebase. The current project identity, however, is **battery/BMS-first**.

---

## Battery-Specific Capabilities

### Multi-model BMS workflow
The repo contains a battery-oriented path for:
- SoH regression
- fault classification
- degradation trend analysis
- ensemble-style prediction flow

### Chemistry-aware data generation
Battery data generation and simulation support battery chemistries such as:
- NMC
- LFP
- NCA

### Dataset handling
The system includes:
- dataset registry
- dataset compatibility checks
- augmentation hooks
- SP20 dataset preprocessing
- battery schema normalization

### Edge deployment thinking
The repo includes export-oriented utilities for:
- joblib / ONNX style packaging
- tiered deployment concepts
- constrained-device deployment discussion

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the battery-first demo
```bash
make bms-demo
```

### 3. Launch dashboard
```bash
make dashboard
```

### 4. Run tests
```bash
make test
```

---

## Recommended Placement Demo

This is the path you should show during placements:

```bash
# 1) Run the battery monitoring demo
python -m src.monitoring.bms_monitoring_loop --demo --batches 10 --chemistry NMC

# 2) Optionally register datasets / models
python -m src.cli.add_dataset --list
python -m src.cli.add_model

# 3) Launch dashboard
python -m src.dashboard.dashboard
```

### What to say while demoing
> “This project focuses on post-deployment ML reliability for battery systems. I built a reusable monitoring backbone and adapted it into a BMS platform supporting dataset registration, multi-model monitoring, drift detection, fleet-health tracking, and deployment-oriented export.”

---

## Project Structure

```text
ml_monitor/
├── src/
│   ├── alerts/                 # Alerting and severity management
│   ├── api/                    # Real-time serving APIs (legacy churn API + extensible serving layer)
│   ├── cli/                    # CLI tools for model and dataset registration
│   ├── dashboard/              # Plotly Dash monitoring UI
│   ├── data/                   # Battery generators, adapters, registries, preprocessors
│   ├── models/                 # BMS models, LSTM wrapper, edge exporter, training utilities
│   └── monitoring/             # Drift detection, monitoring loops, retraining, registry, ROI
├── data/                       # Reference, processed, and dataset assets
├── notebooks/                  # Analysis notebooks
├── scripts/                    # Utility and registry scripts
├── tests/                      # Unit tests and API tests
├── HOW_TO_USE.md               # BMS-first quick usage guide
├── MODEL_CARD.md               # Battery model card
└── README.md
```

---

## Core Components

### Reusable ML monitoring core
These are generic and reusable:
- `src/monitoring/drift_detector.py`
- `src/monitoring/performance_monitor.py`
- `src/monitoring/retrain_pipeline.py`
- `src/alerts/alert_manager.py`
- `src/dashboard/dashboard.py`
- `src/monitoring/model_registry.py`

### Battery application layer
These make the project distinctive:
- `src/data/battery_data_generator.py`
- `src/data/battery_dataset_adapter.py`
- `src/data/dataset_registry.py`
- `src/data/sp20_data_preprocessor.py`
- `src/models/bms_models.py`
- `src/models/soc_lstm_wrapper.py`
- `src/models/edge_exporter.py`
- `src/monitoring/bms_monitoring_loop.py`

---

## Legacy Example Notice

This repository still contains some **legacy tabular/churn-oriented example modules** from the earlier reusable monitoring-core phase of the project.

They are retained for reference and testing purposes, but they are **not the primary identity of the repository**.

For placement discussions, presentations, and demos, you should position this project as:

> **A battery-focused ML monitoring and deployment platform for BMS applications, built on top of a reusable monitoring core.**

---
