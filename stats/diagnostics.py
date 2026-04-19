"""Pre-flight diagnostics for A/B experiments.

Currently supported:
- Sample Ratio Mismatch (SRM) check via chi-square goodness-of-fit test.
  Reference: Fabijan et al., "Diagnosing Sample Ratio Mismatch in
  Online Controlled Experiments", KDD 2019.
"""
import pandas as pd
from scipy import stats


def srm_check(
    df: pd.DataFrame,
    group_col: str,
    expected_split: list[float] | None = None,
    alpha: float = 0.001,
) -> dict:
    """
    Chi-square goodness-of-fit test for Sample Ratio Mismatch.

    Uses alpha = 0.001 by default, following industry convention. SRM checks
    should be strict because false positives are cheap (re-run) but false
    negatives are expensive (ship a bad decision).

    Returns a minimal schema (status + verdict + observed counts only).
    Raw chi-square statistic and p-value are intentionally omitted from
    the return value so that downstream LLM consumers cannot re-derive
    their own judgment at a different threshold.
    """
    counts = df[group_col].value_counts().sort_index()
    groups = counts.index.tolist()
    observed = counts.values.astype(float)
    n = observed.sum()

    if expected_split is None:
        expected_split = [1.0 / len(groups)] * len(groups)

    assert len(expected_split) == len(groups), (
        f"expected_split has {len(expected_split)} entries but data has "
        f"{len(groups)} groups"
    )
    assert abs(sum(expected_split) - 1.0) < 1e-6, "expected_split must sum to 1"

    expected = [p * n for p in expected_split]
    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    srm_detected = bool(p_value < alpha)

    return {
        "check_name": "Sample Ratio Mismatch",
        "status": "FAILED" if srm_detected else "PASSED",
        "verdict": (
            "SRM DETECTED -- randomization may be broken; do NOT trust "
            "downstream test results until root cause is identified."
            if srm_detected
            else "Randomization check passed. Group sizes are consistent "
            "with the expected split. Safe to proceed with primary analysis."
        ),
        "groups": [str(g) for g in groups],
        "observed_counts": [int(c) for c in observed],
    }