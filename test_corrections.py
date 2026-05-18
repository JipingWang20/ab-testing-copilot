"""Unit tests for stats.corrections.

Run with:  pytest test_corrections.py -v
"""
import pytest

from stats.corrections import multiple_comparisons_correction


# ============================================================
# Worked example from teaching notes
# ============================================================

def test_bh_worked_example_matches_teaching():
    """5 p-values [0.003, 0.012, 0.018, 0.041, 0.089] at alpha=0.05.
    Expected: top 3 significant under BH, only top 1 under Bonferroni."""
    tests = [
        {"metric": "m1", "p_value": 0.003},
        {"metric": "m2", "p_value": 0.012},
        {"metric": "m3", "p_value": 0.018},
        {"metric": "m4", "p_value": 0.041},
        {"metric": "m5", "p_value": 0.089},
    ]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    sig = [row["significant"] for row in r["results"]]
    assert sig == [True, True, True, False, False]


def test_bonferroni_only_keeps_strongest_signal():
    tests = [
        {"metric": "m1", "p_value": 0.003},
        {"metric": "m2", "p_value": 0.012},
        {"metric": "m3", "p_value": 0.018},
        {"metric": "m4", "p_value": 0.041},
        {"metric": "m5", "p_value": 0.089},
    ]
    r = multiple_comparisons_correction(tests=tests, method="bonferroni")
    sig = [row["significant"] for row in r["results"]]
    # Only m1 (p=0.003) is below 0.05/5 = 0.01
    assert sig == [True, False, False, False, False]


# ============================================================
# Bonferroni math
# ============================================================

def test_bonferroni_adjusts_by_multiplying_by_m():
    """p_adj = min(p * m, 1) under Bonferroni."""
    tests = [
        {"metric": "a", "p_value": 0.01},
        {"metric": "b", "p_value": 0.02},
        {"metric": "c", "p_value": 0.40},
    ]
    r = multiple_comparisons_correction(tests=tests, method="bonferroni")
    p_adj = [row["p_value_adjusted"] for row in r["results"]]
    assert p_adj[0] == pytest.approx(0.03, abs=1e-9)   # 0.01 * 3
    assert p_adj[1] == pytest.approx(0.06, abs=1e-9)   # 0.02 * 3
    assert p_adj[2] == pytest.approx(1.0,  abs=1e-9)   # capped at 1.0


def test_bh_less_conservative_than_bonferroni():
    """For the same data, BH should call at least as many tests significant."""
    tests = [{"metric": f"m{i}", "p_value": p}
             for i, p in enumerate([0.001, 0.01, 0.02, 0.03, 0.04])]
    r_bh = multiple_comparisons_correction(tests=tests, method="bh")
    r_bonf = multiple_comparisons_correction(tests=tests, method="bonferroni")
    assert r_bh["n_significant_corrected"] >= r_bonf["n_significant_corrected"]


# ============================================================
# Edge cases
# ============================================================

def test_single_test_is_unchanged_under_bh():
    """With m=1, BH should give the same result as the raw test."""
    r = multiple_comparisons_correction(
        tests=[{"metric": "x", "p_value": 0.03}], method="bh"
    )
    assert r["results"][0]["p_value_adjusted"] == pytest.approx(0.03)
    assert r["results"][0]["significant"] is True


def test_single_test_is_unchanged_under_bonferroni():
    r = multiple_comparisons_correction(
        tests=[{"metric": "x", "p_value": 0.03}], method="bonferroni"
    )
    assert r["results"][0]["p_value_adjusted"] == pytest.approx(0.03)
    assert r["results"][0]["significant"] is True


def test_all_highly_significant_all_survive():
    tests = [{"metric": f"m{i}", "p_value": 1e-6} for i in range(10)]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert r["n_significant_corrected"] == 10


def test_all_non_significant_none_survive():
    tests = [{"metric": f"m{i}", "p_value": 0.5} for i in range(10)]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert r["n_significant_corrected"] == 0


# ============================================================
# Monotonicity / sanity invariants
# ============================================================

def test_adjusted_pvalues_at_least_as_large_as_original():
    """For any correction, adjusted p-value >= original."""
    tests = [{"metric": f"m{i}", "p_value": p}
             for i, p in enumerate([0.001, 0.01, 0.02, 0.03, 0.04, 0.1, 0.5])]
    for method in ["bh", "bonferroni"]:
        r = multiple_comparisons_correction(tests=tests, method=method)
        for row in r["results"]:
            assert row["p_value_adjusted"] >= row["p_value"] - 1e-9


def test_corrected_count_at_most_uncorrected_count():
    tests = [{"metric": f"m{i}", "p_value": p}
             for i, p in enumerate([0.001, 0.03, 0.04, 0.06, 0.5])]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert r["n_significant_corrected"] <= r["n_significant_uncorrected"]


def test_input_order_preserved_in_output():
    tests = [
        {"metric": "zzz", "p_value": 0.04},
        {"metric": "aaa", "p_value": 0.001},
        {"metric": "mmm", "p_value": 0.5},
    ]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert [row["metric"] for row in r["results"]] == ["zzz", "aaa", "mmm"]


# ============================================================
# Input validation
# ============================================================

def test_rejects_empty_tests():
    with pytest.raises(ValueError, match="non-empty"):
        multiple_comparisons_correction(tests=[], method="bh")


def test_rejects_invalid_method():
    with pytest.raises(ValueError, match="bh.*bonferroni"):
        multiple_comparisons_correction(
            tests=[{"metric": "x", "p_value": 0.1}], method="holm"
        )


def test_rejects_out_of_range_pvalue():
    with pytest.raises(ValueError, match="p_value.*in"):
        multiple_comparisons_correction(
            tests=[{"metric": "x", "p_value": 1.5}], method="bh"
        )


def test_rejects_malformed_test_dict():
    with pytest.raises(ValueError, match="metric.*p_value"):
        multiple_comparisons_correction(
            tests=[{"metric": "x"}], method="bh"
        )


def test_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha"):
        multiple_comparisons_correction(
            tests=[{"metric": "x", "p_value": 0.1}], method="bh", alpha=1.5
        )


# ============================================================
# Verdict text (the LLM quotes these)
# ============================================================

def test_verdict_reports_test_count():
    tests = [{"metric": f"m{i}", "p_value": 0.5} for i in range(4)]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert "4 tests" in r["verdict"]


def test_verdict_mentions_lost_significance():
    """When correction kills a result, the verdict should say so."""
    tests = [
        {"metric": "m1", "p_value": 0.001},   # survives
        {"metric": "m2", "p_value": 0.04},    # loses under correction
        {"metric": "m3", "p_value": 0.045},   # loses under correction
        {"metric": "m4", "p_value": 0.048},   # loses under correction
    ]
    r = multiple_comparisons_correction(tests=tests, method="bonferroni")
    assert "lost significance" in r["verdict"].lower()


def test_verdict_mentions_no_correction_loss_when_all_survive():
    tests = [{"metric": f"m{i}", "p_value": 1e-6} for i in range(5)]
    r = multiple_comparisons_correction(tests=tests, method="bh")
    assert "survived" in r["verdict"].lower()
