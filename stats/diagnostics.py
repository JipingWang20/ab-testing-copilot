"""Pre-flight diagnostics for A/B experiments.

Currently supported:
- Sample Ratio Mismatch (SRM) check via chi-square goodness-of-fit test.
  Reference: Fabijan et al., "Diagnosing Sample Ratio Mismatch in
  Online Controlled Experiments", KDD 2019.
- CUPED pre-flight check (balance + correlation strength).
"""
import numpy as np
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


def cuped_check(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    covariate: str,
    min_abs_corr: float = 0.1,
    alpha_balance: float = 0.001,
) -> dict:
    """
    Pre-flight check for CUPED variance reduction.

    Decides whether the named pre-period covariate is suitable as a
    control variate by checking two conditions:

    1. Balance: the covariate must not differ significantly across arms.
       A significant pre-treatment difference indicates either broken
       randomization OR that the "pre-period" covariate is contaminated
       by treatment (e.g., captured after treatment started). Either
       case invalidates CUPED. Tested with Welch's t-test at
       alpha = 0.001 by default -- matching SRM strictness.

    2. Strength: |corr(metric, covariate)| must exceed `min_abs_corr`
       (default 0.1). Below this, expected variance reduction is corr^2
       < 1%, and the operational complexity isn't justified.

    Returns one of three verdicts:
    - APPLY:   both checks pass; call two_sample_test with use_cuped=True
    - SKIP:    correlation too weak; run standard test
    - BLOCKED: balance check failed; do NOT use CUPED

    Returns a minimal schema (status + verdict only). Raw p-value,
    correlation, theta, and per-arm covariate means are intentionally
    omitted so the LLM cannot override the verdict by re-deriving
    judgment at a different threshold.
    """
    # ---- validate inputs ----
    for col in (metric, group_col, covariate):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in dataframe.")

    sub = df[[group_col, metric, covariate]].dropna()
    groups = sorted(sub[group_col].unique())
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}: {groups}"
    g1, g2 = groups

    a_cov = sub.loc[sub[group_col] == g1, covariate].values.astype(float)
    b_cov = sub.loc[sub[group_col] == g2, covariate].values.astype(float)
    y = sub[metric].values.astype(float)
    x = sub[covariate].values.astype(float)

    # ---- 1. balance check (run first: leakage can inflate correlation) ----
    var_a = np.var(a_cov, ddof=1) if len(a_cov) > 1 else 0.0
    var_b = np.var(b_cov, ddof=1) if len(b_cov) > 1 else 0.0
    if var_a == 0 and var_b == 0:
        return {
            "check_name": "CUPED Pre-flight",
            "status": "BLOCKED",
            "verdict": (
                f"Covariate {covariate!r} has zero variance within both "
                f"arms. CUPED cannot be applied. Investigate the data "
                f"pipeline before re-running."
            ),
        }

    _, p_balance = stats.ttest_ind(a_cov, b_cov, equal_var=False)
    if p_balance < alpha_balance:
        return {
            "check_name": "CUPED Pre-flight",
            "status": "BLOCKED",
            "verdict": (
                f"Covariate {covariate!r} differs significantly across arms "
                f"before treatment, suggesting either broken randomization "
                f"or post-treatment contamination of the 'pre-period' field. "
                f"Do NOT use CUPED -- the adjustment would distort results. "
                f"Investigate the data pipeline before re-running."
            ),
        }

    # ---- 2. correlation strength ----
    if np.var(x, ddof=1) == 0 or np.var(y, ddof=1) == 0:
        return {
            "check_name": "CUPED Pre-flight",
            "status": "SKIP",
            "verdict": (
                f"Covariate {covariate!r} or metric {metric!r} has zero "
                f"variance overall. Run the standard test without "
                f"variance reduction."
            ),
        }

    corr = float(np.corrcoef(y, x)[0, 1])
    if abs(corr) < min_abs_corr:
        return {
            "check_name": "CUPED Pre-flight",
            "status": "SKIP",
            "verdict": (
                f"Covariate {covariate!r} is too weakly correlated with "
                f"{metric!r} to justify CUPED (expected variance reduction "
                f"below 1%). Run the standard test without variance "
                f"reduction."
            ),
        }

    # ---- 3. apply ----
    expected_vr = corr ** 2
    return {
        "check_name": "CUPED Pre-flight",
        "status": "APPLY",
        "verdict": (
            f"Pre-period covariate {covariate!r} is suitable for CUPED "
            f"variance reduction (expected reduction approximately "
            f"{expected_vr:.0%}). Proceed with use_cuped=True and "
            f"covariate={covariate!r}."
        ),
    }