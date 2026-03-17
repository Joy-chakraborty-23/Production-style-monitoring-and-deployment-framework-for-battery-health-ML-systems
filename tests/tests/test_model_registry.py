"""
tests/test_model_registry.py — Unit tests for ModelRegistry

Run with: pytest tests/test_model_registry.py -v
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.model_registry import ModelRegistry, DEFAULT_THRESHOLDS, STATUS_HEALTHY, STATUS_WARNING, STATUS_CRITICAL


@pytest.fixture
def registry(tmp_path):
    """Fresh in-memory (temp file) registry for each test."""
    db_path = tmp_path / "test_registry.db"
    return ModelRegistry(f"sqlite:///{db_path}")


class TestRegistration:
    def test_register_new_model_returns_id(self, registry):
        mid = registry.register_model(name="test_model", version="1.0.0")
        assert isinstance(mid, int)
        assert mid > 0

    def test_register_same_model_twice_returns_same_id(self, registry):
        id1 = registry.register_model("duplicate", "1.0.0")
        id2 = registry.register_model("duplicate", "1.0.0")
        assert id1 == id2

    def test_register_different_versions_returns_different_ids(self, registry):
        id1 = registry.register_model("versioned", "1.0.0")
        id2 = registry.register_model("versioned", "2.0.0")
        assert id1 != id2

    def test_default_thresholds_applied(self, registry):
        registry.register_model("thresh_model", "1.0.0")
        thresholds = registry._get_thresholds(1)
        assert thresholds["auc_warning"]  == DEFAULT_THRESHOLDS["auc_warning"]
        assert thresholds["auc_critical"] == DEFAULT_THRESHOLDS["auc_critical"]

    def test_custom_thresholds_override_defaults(self, registry):
        registry.register_model(
            "custom", "1.0.0",
            thresholds={"auc_warning": 0.75, "auc_critical": 0.65}
        )
        thresholds = registry._get_thresholds(1)
        assert thresholds["auc_warning"]  == 0.75
        assert thresholds["auc_critical"] == 0.65

    def test_list_models_returns_dataframe(self, registry):
        registry.register_model("model_a", "1.0.0")
        registry.register_model("model_b", "1.0.0")
        df = registry.list_models()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns


class TestBatchMetrics:
    def test_log_batch_metrics_single_batch(self, registry):
        mid = registry.register_model("churn", "1.0.0")
        registry.log_batch_metrics(mid, batch_id=0, auc_roc=0.87, f1_score=0.79)
        df = registry.get_model_history(mid)
        assert len(df) == 1
        assert abs(df.iloc[0]["auc_roc"] - 0.87) < 1e-4

    def test_log_multiple_batches_ordered_by_batch_id(self, registry):
        mid = registry.register_model("churn", "1.0.0")
        for i, auc in enumerate([0.87, 0.84, 0.81, 0.73]):
            registry.log_batch_metrics(mid, batch_id=i, auc_roc=auc)
        df = registry.get_model_history(mid)
        assert len(df) == 4
        assert list(df["batch_id"]) == [0, 1, 2, 3]

    def test_status_updates_to_warning_when_auc_below_warning(self, registry):
        mid = registry.register_model(
            "churn", "1.0.0",
            thresholds={"auc_warning": 0.80, "auc_critical": 0.70}
        )
        registry.log_batch_metrics(mid, batch_id=0, auc_roc=0.78)
        models = registry.list_models()
        assert models.iloc[0]["status"] == STATUS_WARNING

    def test_status_updates_to_critical_when_auc_below_critical(self, registry):
        mid = registry.register_model(
            "churn", "1.0.0",
            thresholds={"auc_warning": 0.80, "auc_critical": 0.70}
        )
        registry.log_batch_metrics(mid, batch_id=0, auc_roc=0.65)
        models = registry.list_models()
        assert models.iloc[0]["status"] == STATUS_CRITICAL

    def test_status_healthy_when_auc_above_warning(self, registry):
        mid = registry.register_model(
            "churn", "1.0.0",
            thresholds={"auc_warning": 0.80, "auc_critical": 0.70}
        )
        registry.log_batch_metrics(mid, batch_id=0, auc_roc=0.88)
        models = registry.list_models()
        assert models.iloc[0]["status"] == STATUS_HEALTHY

    def test_last_n_batches_filter(self, registry):
        mid = registry.register_model("churn", "1.0.0")
        for i in range(10):
            registry.log_batch_metrics(mid, batch_id=i, auc_roc=0.85)
        df = registry.get_model_history(mid, last_n_batches=3)
        assert len(df) == 3


class TestDriftDetails:
    def test_log_and_retrieve_drift_details(self, registry):
        from unittest.mock import MagicMock
        mid = registry.register_model("churn", "1.0.0")

        # Create mock DriftResult objects
        results = []
        for feat, score, drifted in [
            ("monthly_charges", 0.25, True),
            ("tenure",          0.05, False),
        ]:
            r = MagicMock()
            r.feature    = feat
            r.method     = "PSI+KS"
            r.score      = score
            r.p_value    = 0.01 if drifted else 0.45
            r.severity   = "critical" if drifted else "none"
            r.is_drifted = drifted
            results.append(r)

        registry.log_drift_details(mid, batch_id=5, drift_results=results)
        df = registry.get_model_drift_history(mid, batch_id=5)
        assert len(df) == 2
        assert set(df["feature"]) == {"monthly_charges", "tenure"}

    def test_drift_details_filter_by_batch(self, registry):
        from unittest.mock import MagicMock
        mid = registry.register_model("churn", "1.0.0")

        for batch_id in [1, 2, 3]:
            r = MagicMock()
            r.feature = "tenure"; r.method = "PSI"; r.score = 0.05
            r.p_value = 0.3; r.severity = "none"; r.is_drifted = False
            registry.log_drift_details(mid, batch_id=batch_id, drift_results=[r])

        df_all    = registry.get_model_drift_history(mid)
        df_single = registry.get_model_drift_history(mid, batch_id=2)
        assert len(df_all) == 3
        assert len(df_single) == 1


class TestRetrainLog:
    def test_log_and_retrieve_retrain_event(self, registry):
        mid = registry.register_model("churn", "1.0.0")
        registry.log_retrain_event(
            model_id=mid, trigger_batch=8,
            trigger_reason="AUC below critical threshold",
            old_auc=0.69, new_auc=0.84,
            promoted=True, promotion_reason="Improved by 0.15"
        )
        df = registry.get_retrain_history(mid)
        assert len(df) == 1
        assert abs(df.iloc[0]["old_auc"] - 0.69) < 1e-4
        assert df.iloc[0]["promoted"] == 1

    def test_multiple_retrain_events_ordered(self, registry):
        mid = registry.register_model("churn", "1.0.0")
        for batch, old, new in [(8, 0.69, 0.83), (12, 0.77, 0.85)]:
            registry.log_retrain_event(
                mid, batch, "trigger", old, new, True, "ok"
            )
        df = registry.get_retrain_history(mid)
        assert len(df) == 2
        assert df.iloc[0]["trigger_batch"] == 8


class TestFleetHealth:
    def test_fleet_health_returns_dataframe(self, registry):
        for name in ["churn", "ltv", "propensity"]:
            mid = registry.register_model(name, "1.0.0")
            registry.log_batch_metrics(mid, 0, auc_roc=0.85)
        df = registry.get_fleet_health()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "latest_auc" in df.columns

    def test_fleet_health_empty_without_batches(self, registry):
        registry.register_model("empty_model", "1.0.0")
        df = registry.get_fleet_health()
        # Model row exists but latest_auc is null
        assert len(df) == 1
        assert pd.isna(df.iloc[0].get("latest_auc"))

    def test_export_fleet_json(self, registry, tmp_path):
        mid = registry.register_model("churn", "1.0.0")
        registry.log_batch_metrics(mid, 0, auc_roc=0.85)
        out = registry.export_fleet_json(tmp_path / "fleet.json")
        assert out.exists()
        df = pd.read_json(out)
        assert len(df) == 1
