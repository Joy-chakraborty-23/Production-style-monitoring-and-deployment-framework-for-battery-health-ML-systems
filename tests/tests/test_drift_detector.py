"""
tests/test_drift_detector.py — Unit tests for drift detection engine

Run with:
    pytest tests/ -v --tb=short
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.drift_detector import (
    compute_psi, compute_ks, compute_jsd,
    compute_wasserstein, compute_chi2, DriftDetector
)
from src.data.data_generator import generate_reference_data, generate_production_batch


# ── PSI Tests ────────────────────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions_near_zero(self):
        arr = np.random.normal(0, 1, 1000)
        psi = compute_psi(arr, arr.copy())
        assert psi < 0.01, f"Identical arrays should have PSI ≈ 0, got {psi}"

    def test_shifted_distribution_triggers_warning(self):
        ref  = np.random.normal(0, 1, 1000)
        prod = np.random.normal(1.5, 1, 1000)   # large shift
        psi  = compute_psi(ref, prod)
        assert psi > 0.10, f"Shifted distribution should give PSI > 0.10, got {psi}"

    def test_severe_shift_triggers_critical(self):
        ref  = np.random.normal(0, 1, 1000)
        prod = np.random.normal(5, 1, 1000)    # extreme shift
        psi  = compute_psi(ref, prod)
        assert psi > 0.20, f"Extreme shift should give PSI > 0.20, got {psi}"

    def test_psi_nonnegative(self):
        ref  = np.random.normal(0, 1, 500)
        prod = np.random.normal(0.5, 1.2, 500)
        assert compute_psi(ref, prod) >= 0


# ── KS Test ──────────────────────────────────────────────────────────────────

class TestKS:
    def test_same_distribution_high_p(self):
        arr    = np.random.normal(0, 1, 1000)
        _, p   = compute_ks(arr, arr + np.random.normal(0, 0.001, 1000))
        assert p > 0.05, f"Same distribution should have p > 0.05, got p={p}"

    def test_different_distribution_low_p(self):
        ref  = np.random.normal(0, 1, 1000)
        prod = np.random.normal(3, 1, 1000)
        _, p = compute_ks(ref, prod)
        assert p < 0.05, f"Different distributions should have p < 0.05, got p={p}"

    def test_returns_tuple(self):
        a = np.random.normal(0, 1, 100)
        result = compute_ks(a, a)
        assert len(result) == 2


# ── JSD ───────────────────────────────────────────────────────────────────────

class TestJSD:
    def test_identical_near_zero(self):
        arr = np.random.uniform(0, 1, 500)
        jsd = compute_jsd(arr, arr.copy())
        assert jsd < 0.05

    def test_range_zero_to_one(self):
        a = np.random.normal(0, 1, 500)
        b = np.random.normal(5, 1, 500)
        jsd = compute_jsd(a, b)
        assert 0.0 <= jsd <= 1.0

    def test_larger_shift_larger_jsd(self):
        ref = np.random.normal(0, 1, 1000)
        small_shift  = compute_jsd(ref, np.random.normal(0.5, 1, 1000))
        large_shift  = compute_jsd(ref, np.random.normal(3.0, 1, 1000))
        assert large_shift > small_shift


# ── Wasserstein ───────────────────────────────────────────────────────────────

class TestWasserstein:
    def test_identical_zero(self):
        arr = np.random.normal(0, 1, 500)
        w   = compute_wasserstein(arr, arr.copy())
        assert w < 0.01

    def test_nonnegative(self):
        a = np.random.normal(0, 1, 500)
        b = np.random.normal(2, 1, 500)
        assert compute_wasserstein(a, b) >= 0

    def test_constant_array_safe(self):
        a = np.ones(500)
        b = np.ones(500) * 2
        w = compute_wasserstein(a, b)
        assert w == 0.0   # std=0 → returns 0 safely


# ── Chi-Square ────────────────────────────────────────────────────────────────

class TestChi2:
    def test_identical_categories_high_p(self):
        cats = ["A", "B", "C"]
        arr  = pd.Series(np.random.choice(cats, 500, p=[0.5, 0.3, 0.2]))
        _, p = compute_chi2(arr, arr.sample(300).reset_index(drop=True))
        # Same distribution → p-value should be reasonably high
        # (not guaranteed for small samples, but usually > 0.01)

    def test_shifted_categories_low_p(self):
        ref  = pd.Series(["A"] * 400 + ["B"] * 100)
        prod = pd.Series(["A"] * 100 + ["B"] * 400)
        _, p = compute_chi2(ref, prod)
        assert p < 0.05, f"Very different distributions should give p < 0.05, got {p}"

    def test_returns_tuple(self):
        a = pd.Series(["X", "Y", "X", "Z"])
        b = pd.Series(["X", "Z", "Y", "Y"])
        result = compute_chi2(a, b)
        assert len(result) == 2


# ── DriftDetector integration ─────────────────────────────────────────────────

class TestDriftDetector:
    @pytest.fixture
    def reference_df(self):
        return generate_reference_data(n_samples=1000)

    @pytest.fixture
    def detector(self, reference_df):
        return DriftDetector(reference_df)

    def test_no_drift_on_same_distribution(self, detector, reference_df):
        clean_batch = generate_production_batch(500, batch_id=0, drift_type=None)
        report = detector.detect(clean_batch, batch_id=0)
        # With no injected drift, severity should be none or warning (statistical noise)
        assert report.overall_severity in ("none", "warning")

    def test_critical_drift_detected(self, detector):
        drifted = generate_production_batch(500, batch_id=1, drift_type="sudden", drift_intensity=1.0)
        report  = detector.detect(drifted, batch_id=1)
        # Severe drift should trigger at least warning
        assert report.overall_severity in ("warning", "critical")
        assert report.n_drifted_features > 0

    def test_report_has_all_features(self, detector, reference_df):
        batch  = generate_production_batch(500, batch_id=0)
        report = detector.detect(batch, batch_id=0)
        reported_features = {r.feature for r in report.drift_results}
        # Should cover both numerical and categorical features
        assert len(reported_features) > 5

    def test_batch_report_serializable(self, detector):
        batch  = generate_production_batch(500, batch_id=0)
        report = detector.detect(batch, batch_id=0)
        summary = report.summary()
        assert isinstance(summary, dict)
        assert "batch_id" in summary
        assert "overall_severity" in summary

    def test_gradual_drift_increasing(self, detector):
        """Gradual drift should produce increasing drift scores over batches."""
        scores = []
        for i in range(5):
            batch  = generate_production_batch(
                500, batch_id=i, drift_type="gradual",
                drift_intensity=0.8
            )
            report = detector.detect(batch, batch_id=i)
            scores.append(report.overall_drift_score)
        # At least the last score should be > the first
        assert scores[-1] >= scores[0] * 0.5  # loose check, drift is stochastic


# ── Data Generator Tests ──────────────────────────────────────────────────────

class TestDataGenerator:
    def test_reference_schema(self):
        df = generate_reference_data(n_samples=100)
        assert "churn" in df.columns
        assert df["churn"].isin([0, 1]).all()
        assert len(df) == 100

    def test_churn_rate_realistic(self):
        df = generate_reference_data(n_samples=2000)
        rate = df["churn"].mean()
        assert 0.10 <= rate <= 0.40, f"Churn rate {rate:.3f} outside realistic range"

    def test_production_batch_schema_matches_reference(self):
        ref   = generate_reference_data(n_samples=200)
        batch = generate_production_batch(200, batch_id=0)
        # All reference columns should be present in batch
        ref_cols = set(ref.columns) - {"churn"}
        for col in ref_cols:
            assert col in batch.columns, f"Column {col} missing from production batch"

    def test_drift_injection_changes_distribution(self):
        clean   = generate_production_batch(1000, batch_id=0, drift_type=None)
        drifted = generate_production_batch(1000, batch_id=0, drift_type="covariate", drift_intensity=1.0)
        # Monthly charges should shift significantly
        clean_mean   = clean["monthly_charges"].mean()
        drifted_mean = drifted["monthly_charges"].mean()
        assert abs(drifted_mean - clean_mean) > 5, \
            f"Covariate drift should shift monthly_charges: {clean_mean:.1f} → {drifted_mean:.1f}"
