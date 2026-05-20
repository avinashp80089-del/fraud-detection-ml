"""Unit tests for fraud detection pipeline."""
import numpy as np
import pytest

from src.data_pipeline import engineer_features, generate_sample_data, handle_class_imbalance, FEATURE_COLS
from src.ab_testing import compute_sample_size, run_proportion_test, run_metric_ttest
from src.evaluation import evaluate_at_threshold, calibrate_threshold, _population_stability_index


# ── Data pipeline ────────────────────────────────────────────────────────────

def test_generate_sample_data():
    df = generate_sample_data(n_samples=1_000)
    assert len(df) == 1_000
    assert "is_fraud" in df.columns
    assert df["amount"].gt(0).all()
    assert df["is_fraud"].isin([0, 1]).all()


def test_engineer_features_columns():
    df = generate_sample_data(n_samples=500)
    df_feat = engineer_features(df)
    for col in ["amount_log", "txn_count_1h", "hour_of_day", "is_weekend", "velocity_x_amount"]:
        assert col in df_feat.columns, f"Missing feature: {col}"


def test_handle_class_imbalance():
    rng = np.random.RandomState(0)
    X = rng.randn(1000, 5)
    y = np.zeros(1000, dtype=int)
    y[:3] = 1  # very imbalanced
    X_res, y_res = handle_class_imbalance(X, y)
    assert y_res.sum() > y.sum(), "SMOTE should have created more positive samples"
    assert len(X_res) == len(y_res)


# ── A/B testing ──────────────────────────────────────────────────────────────

def test_compute_sample_size():
    n = compute_sample_size(baseline_rate=0.003, minimum_detectable_effect=0.001)
    assert n > 0
    assert isinstance(n, int)


def test_run_proportion_test_significant():
    result = run_proportion_test(
        control_conversions=30, control_n=10_000,
        treatment_conversions=55, treatment_n=10_000,
    )
    assert result["significant"] is True
    assert result["absolute_lift"] > 0


def test_run_proportion_test_not_significant():
    result = run_proportion_test(
        control_conversions=30, control_n=10_000,
        treatment_conversions=31, treatment_n=10_000,
    )
    assert result["significant"] is False


def test_run_metric_ttest():
    rng = np.random.RandomState(42)
    control = rng.normal(0.80, 0.02, 100)
    treatment = rng.normal(0.85, 0.02, 100)
    result = run_metric_ttest(control, treatment)
    assert result["significant"] is True
    assert result["delta"] > 0


# ── Evaluation ───────────────────────────────────────────────────────────────

def test_evaluate_at_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = evaluate_at_threshold(y, p, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_calibrate_threshold_precision_target():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.05).astype(int)
    p = rng.rand(500)
    p[y == 1] += 0.3
    p = p.clip(0, 1)
    threshold, metrics = calibrate_threshold(y, p, target_precision=0.5)
    assert 0 < threshold < 1


def test_population_stability_index_same_dist():
    rng = np.random.RandomState(0)
    x = rng.randn(1000)
    psi = _population_stability_index(x, x)
    assert psi < 0.1  # same distribution → low PSI
