"""Unit tests for stats.tests and stats.diagnostics.

Run with:  pytest test_stats.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats.tests import _cuped_adjust, two_sample_test
from stats.diagnostics import cuped_check, srm_check


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_strong_cov():
    """y = 0.6*x + 2*I(B) + noise. Covariate explains ~40% of variance."""
    rng = np.random.default_rng(0)
    n = 2000
    group = rng.choice(["A", "B"], size=n)
    pre = rng.normal(100, 20, n)
    outcome = 0.6 * pre + np.where(group == "B", 2.0, 0) + rng.normal(0, 15, n)
    return pd.DataFrame({"group": group, "revenue": outcome, "pre_revenue": pre})


@pytest.fixture
def synthetic_weak_cov():
    """Covariate is independent of outcome. CUPED should be a near no-op."""
    rng = np.random.default_rng(1)
    n = 2000
    group = rng.choice(["A", "B"], size=n)
    pre = rng.normal(100, 20, n)                       # independent of outcome
    outcome = np.where(group == "B", 2.0, 0) + rng.normal(0, 15, n)
    return pd.DataFrame({"group": group, "revenue": outcome, "pre_revenue": pre})


@pytest.fixture
def synthetic_leaky_cov():
    """Covariate artificially shifted in arm B. Simulates leakage."""
    rng = np.random.default_rng(2)
    n = 2000
    group = rng.choice(["A", "B"], size=n)
    pre = rng.normal(100, 20, n) + np.where(group == "B", 5.0, 0)  # contamination
    outcome = 0.6 * pre + np.where(group == "B", 2.0, 0) + rng.normal(0, 15, n)
    return pd.DataFrame({"group": group, "revenue": outcome, "pre_revenue": pre})


@pytest.fixture
def cookie_cats_df():
    """Real dataset. Skipped if missing so CI doesn't fail without it."""
    path = Path("data/cookie_cats.csv")
    if not path.exists():
        pytest.skip(f"{path} not found")
    return pd.read_csv(path)


# ============================================================
# _cuped_adjust  (low-level math helper)
# ============================================================

def test_cuped_adjust_strong_covariate_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = 0.7 * x + rng.normal(0, 1, 5000)               # corr ≈ 0.57
    _, theta, vr = _cuped_adjust(y, x)
    assert 0.5 < theta < 0.9                           # near true 0.7
    assert vr > 0.25                                   # ~33% expected


def test_cuped_adjust_zero_correlation_yields_no_reduction():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)                         # independent of x
    _, theta, vr = _cuped_adjust(y, x)
    assert abs(theta) < 0.1
    assert abs(vr) < 0.02                              # near zero


def test_cuped_adjust_constant_covariate_is_identity():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([5.0, 5.0, 5.0, 5.0])                 # var(x) = 0
    y_adj, theta, vr = _cuped_adjust(y, x)
    np.testing.assert_array_equal(y_adj, y)
    assert theta == 0.0
    assert vr == 0.0


def test_cuped_adjust_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        _cuped_adjust(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# ============================================================
# two_sample_test  with CUPED
# ============================================================

def test_two_sample_test_cuped_narrows_ci(synthetic_strong_cov):
    r_naive = two_sample_test(
        synthetic_strong_cov, "revenue", "group", "continuous"
    )
    r_cuped = two_sample_test(
        synthetic_strong_cov, "revenue", "group", "continuous",
        use_cuped=True, covariate="pre_revenue",
    )
    width_naive = r_naive["ci_95"][1] - r_naive["ci_95"][0]
    width_cuped = r_cuped["ci_95"][1] - r_cuped["ci_95"][0]
    # CI width scales with sqrt(variance), so ~40% variance reduction
    # implies ~22% width reduction. Demand at least 15% to leave headroom.
    assert width_cuped < width_naive * 0.85


def test_two_sample_test_cuped_reports_variance_reduction(synthetic_strong_cov):
    r = two_sample_test(
        synthetic_strong_cov, "revenue", "group", "continuous",
        use_cuped=True, covariate="pre_revenue",
    )
    assert r["cuped_applied"] is True
    assert r["variance_reduction"] is not None
    assert r["variance_reduction"] > 0.25                # strong covariate


def test_two_sample_test_without_cuped_omits_variance_reduction(synthetic_strong_cov):
    r = two_sample_test(synthetic_strong_cov, "revenue", "group", "continuous")
    assert r["cuped_applied"] is False
    assert r["variance_reduction"] is None


def test_two_sample_test_cuped_unbiased_when_covariate_is_noise(synthetic_weak_cov):
    """With an independent covariate, CUPED should not meaningfully move
    the point estimate. Confirms unbiasedness."""
    r_naive = two_sample_test(
        synthetic_weak_cov, "revenue", "group", "continuous"
    )
    r_cuped = two_sample_test(
        synthetic_weak_cov, "revenue", "group", "continuous",
        use_cuped=True, covariate="pre_revenue",
    )
    diff = abs(r_naive["absolute_effect"] - r_cuped["absolute_effect"])
    assert diff < 0.1                                  # essentially identical
    assert abs(r_cuped["variance_reduction"]) < 0.02


def test_two_sample_test_cuped_rejects_binary_metric(synthetic_strong_cov):
    df = synthetic_strong_cov.copy()
    df["converted"] = (df["revenue"] > df["revenue"].median()).astype(int)
    with pytest.raises(ValueError, match="continuous"):
        two_sample_test(
            df, "converted", "group", "binary",
            use_cuped=True, covariate="pre_revenue",
        )


def test_two_sample_test_cuped_requires_covariate(synthetic_strong_cov):
    with pytest.raises(ValueError, match="covariate"):
        two_sample_test(
            synthetic_strong_cov, "revenue", "group", "continuous",
            use_cuped=True,
        )


def test_two_sample_test_cuped_rejects_missing_covariate_column(synthetic_strong_cov):
    with pytest.raises(ValueError, match="not in dataframe"):
        two_sample_test(
            synthetic_strong_cov, "revenue", "group", "continuous",
            use_cuped=True, covariate="nonexistent_column",
        )


# ============================================================
# cuped_check  (three branches)
# ============================================================

def test_cuped_check_apply_on_strong_balanced_covariate(synthetic_strong_cov):
    r = cuped_check(synthetic_strong_cov, "revenue", "group", "pre_revenue")
    assert r["status"] == "APPLY"
    assert "use_cuped=True" in r["verdict"]


def test_cuped_check_skip_on_weak_covariate(synthetic_weak_cov):
    r = cuped_check(synthetic_weak_cov, "revenue", "group", "pre_revenue")
    assert r["status"] == "SKIP"
    assert "weakly correlated" in r["verdict"]


def test_cuped_check_blocked_on_leaky_covariate(synthetic_leaky_cov):
    r = cuped_check(synthetic_leaky_cov, "revenue", "group", "pre_revenue")
    assert r["status"] == "BLOCKED"
    assert "differs significantly" in r["verdict"]


def test_cuped_check_rejects_missing_column(synthetic_strong_cov):
    with pytest.raises(ValueError, match="not in dataframe"):
        cuped_check(synthetic_strong_cov, "revenue", "group", "ghost_column")


def test_cuped_check_blocked_on_zero_variance_covariate():
    df = pd.DataFrame({
        "group": ["A"] * 100 + ["B"] * 100,
        "revenue": np.random.default_rng(0).normal(0, 1, 200),
        "pre_revenue": np.zeros(200),
    })
    r = cuped_check(df, "revenue", "group", "pre_revenue")
    assert r["status"] == "BLOCKED"


# ============================================================
# Real-data smoke tests  (preserved from original script)
# ============================================================

def test_cookie_cats_binary_returns_expected_schema(cookie_cats_df):
    r = two_sample_test(
        cookie_cats_df, metric="retention_7",
        group_col="version", metric_type="binary",
    )
    expected_keys = {
        "group_a", "group_b", "n_a", "n_b", "mean_a", "mean_b",
        "absolute_effect", "p_value", "ci_95", "significant",
        "cuped_applied", "variance_reduction",
    }
    assert expected_keys.issubset(r.keys())
    assert r["cuped_applied"] is False
    assert 0 <= r["p_value"] <= 1


def test_cookie_cats_continuous_returns_expected_schema(cookie_cats_df):
    r = two_sample_test(
        cookie_cats_df, metric="sum_gamerounds",
        group_col="version", metric_type="continuous",
    )
    assert 0 <= r["p_value"] <= 1
    assert r["n_a"] > 0 and r["n_b"] > 0


# ============================================================
# SRM regression check  (existing behavior shouldn't break)
# ============================================================

def test_srm_check_passes_on_balanced_data(synthetic_strong_cov):
    r = srm_check(synthetic_strong_cov, "group")
    assert r["status"] == "PASSED"


def test_srm_check_fails_on_imbalanced_data():
    df = pd.DataFrame({"group": ["A"] * 1500 + ["B"] * 500})
    r = srm_check(df, "group")
    assert r["status"] == "FAILED"