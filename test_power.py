"""Unit tests for stats.power.

Run with:  pytest test_power.py -v
"""
import numpy as np
import pandas as pd
import pytest

from stats.power import power_analysis


# ============================================================
# Continuous: solve for n
# ============================================================

def test_continuous_solve_n_basic():
    r = power_analysis(metric_type="continuous", solve_for="n",
                       baseline_std=19, mde=1.0)
    # Rule of thumb: n ≈ 16σ²/MDE² = 16*361 = 5776. Statsmodels exact: ~5668.
    assert 5000 < r["n_per_arm"] < 6500
    assert r["n_total"] == 2 * r["n_per_arm"]
    assert r["solved_for"] == "n_per_arm"


def test_continuous_solve_n_smaller_mde_needs_more_users():
    r_big = power_analysis(metric_type="continuous", solve_for="n",
                           baseline_std=10, mde=2.0)
    r_small = power_analysis(metric_type="continuous", solve_for="n",
                             baseline_std=10, mde=1.0)
    # Halving MDE should ~quadruple sample size
    assert 3.5 < r_small["n_per_arm"] / r_big["n_per_arm"] < 4.5


def test_continuous_solve_n_from_column():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"pre_revenue": rng.normal(100, 19, 5000)})
    r = power_analysis(df=df, metric_type="continuous", solve_for="n",
                       baseline_column="pre_revenue", mde=1.0)
    assert 18 < r["baseline_std"] < 20            # close to true σ=19
    assert 5000 < r["n_per_arm"] < 6500


def test_continuous_solve_n_requires_mde():
    with pytest.raises(ValueError, match="requires mde"):
        power_analysis(metric_type="continuous", solve_for="n",
                       baseline_std=19)


def test_continuous_solve_n_requires_std_source():
    with pytest.raises(ValueError, match="baseline_std or baseline_column"):
        power_analysis(metric_type="continuous", solve_for="n", mde=1.0)


# ============================================================
# Continuous: solve for mde and power
# ============================================================

def test_continuous_solve_mde():
    r = power_analysis(metric_type="continuous", solve_for="mde",
                       baseline_std=19, n_per_arm=2000)
    # With n=2000/arm, std=19, expected MDE ≈ 1.68
    assert 1.5 < r["mde"] < 1.9


def test_continuous_solve_power_matches_empirical():
    """Validate against the 500-seed simulation: n=1000/arm, σ≈19, MDE=2.0
    should give power ≈ 64-65%."""
    r = power_analysis(metric_type="continuous", solve_for="power",
                       baseline_std=19, mde=2.0, n_per_arm=1000)
    assert 0.60 < r["power"] < 0.70


def test_continuous_cuped_increases_power():
    """40% variance reduction = std multiplier of sqrt(0.6) ≈ 0.775."""
    r_naive = power_analysis(metric_type="continuous", solve_for="power",
                             baseline_std=19, mde=2.0, n_per_arm=1000)
    r_cuped = power_analysis(metric_type="continuous", solve_for="power",
                             baseline_std=19 * 0.775, mde=2.0, n_per_arm=1000)
    assert r_cuped["power"] > r_naive["power"] + 0.15  # at least 15pp lift


# ============================================================
# Binary: solve for n, mde, power
# ============================================================

def test_binary_solve_n_basic():
    r = power_analysis(metric_type="binary", solve_for="n",
                       baseline_rate=0.10, mde=0.02)
    # 10% -> 12% lift, 80% power, α=0.05 → ~3800 per arm
    assert 3000 < r["n_per_arm"] < 4500
    assert r["treatment_rate"] == pytest.approx(0.12, abs=1e-6)


def test_binary_solve_n_rare_event_needs_more_users():
    r_common = power_analysis(metric_type="binary", solve_for="n",
                              baseline_rate=0.50, mde=0.02)
    r_rare = power_analysis(metric_type="binary", solve_for="n",
                            baseline_rate=0.05, mde=0.02)
    # Rare events have lower variance per user, but for the same absolute
    # MDE the variance ratio is p(1-p), which is lower for p=0.05 than 0.5.
    # So rare events should need FEWER users for the same absolute lift.
    assert r_rare["n_per_arm"] < r_common["n_per_arm"]


def test_binary_solve_mde():
    r = power_analysis(metric_type="binary", solve_for="mde",
                       baseline_rate=0.10, n_per_arm=2000)
    # With n=2000/arm and 10% baseline, smallest detectable lift ≈ +2.7pp
    assert 0.02 < r["mde"] < 0.035


def test_binary_solve_power():
    r = power_analysis(metric_type="binary", solve_for="power",
                       baseline_rate=0.10, mde=0.02, n_per_arm=2000)
    # Underpowered: only ~50% power with this setup
    assert 0.4 < r["power"] < 0.6


def test_binary_requires_baseline_rate():
    with pytest.raises(ValueError, match="baseline_rate"):
        power_analysis(metric_type="binary", solve_for="n", mde=0.02)


def test_binary_rejects_out_of_range_baseline_rate():
    with pytest.raises(ValueError, match="must be in"):
        power_analysis(metric_type="binary", solve_for="n",
                       baseline_rate=1.5, mde=0.02)


def test_binary_rejects_mde_exceeding_unit_interval():
    """baseline_rate + mde must stay in (0, 1)."""
    with pytest.raises(ValueError, match="must be in"):
        power_analysis(metric_type="binary", solve_for="n",
                       baseline_rate=0.95, mde=0.20)   # would push to 1.15


# ============================================================
# Input validation
# ============================================================

def test_rejects_invalid_solve_for():
    with pytest.raises(ValueError, match="solve_for"):
        power_analysis(metric_type="continuous", solve_for="nonsense",
                       baseline_std=19, mde=1.0)


def test_rejects_invalid_metric_type():
    with pytest.raises(ValueError, match="metric_type"):
        power_analysis(metric_type="ordinal", solve_for="n",
                       baseline_rate=0.1, mde=0.02)


def test_rejects_missing_column_for_baseline():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="not in dataframe"):
        power_analysis(df=df, metric_type="continuous", solve_for="n",
                       baseline_column="missing", mde=1.0)


# ============================================================
# Verdict text checks (the LLM quotes these)
# ============================================================

def test_verdict_contains_key_numbers_continuous():
    r = power_analysis(metric_type="continuous", solve_for="n",
                       baseline_std=19, mde=1.0)
    assert str(r["n_per_arm"]) in r["verdict"] or f"{r['n_per_arm']:,}" in r["verdict"]
    assert "per arm" in r["verdict"]


def test_verdict_flags_underpowered_plan():
    r = power_analysis(metric_type="continuous", solve_for="power",
                       baseline_std=19, mde=0.5, n_per_arm=500)
    assert "underpowered" in r["verdict"].lower()


def test_verdict_confirms_adequate_power():
    r = power_analysis(metric_type="continuous", solve_for="power",
                       baseline_std=10, mde=5.0, n_per_arm=200)
    assert "adequately powered" in r["verdict"].lower()
