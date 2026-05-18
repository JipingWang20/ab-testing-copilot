"""Two-sample statistical tests for A/B experiments."""
import numpy as np
import pandas as pd
from scipy import stats


def _cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Apply CUPED variance reduction to outcome y using pre-experiment covariate x.

    Returns:
        y_adj: CUPED-adjusted outcome, same shape as y
        theta: the regression coefficient used (cov(y,x) / var(x))
        variance_reduction: 1 - var(y_adj) / var(y), in [0, 1]
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.shape != x.shape:
        raise ValueError(f"y and x must have the same shape, got {y.shape} vs {x.shape}")

    var_x = np.var(x, ddof=1)
    if var_x == 0:
        # Covariate is constant -- CUPED degenerates to identity
        return y.copy(), 0.0, 0.0

    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    y_adj = y - theta * (x - np.mean(x))

    var_y = np.var(y, ddof=1)
    variance_reduction = 1.0 - (np.var(y_adj, ddof=1) / var_y) if var_y > 0 else 0.0

    return y_adj, theta, variance_reduction


def two_sample_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    metric_type: str = "continuous",
    alpha: float = 0.05,
    use_cuped: bool = False,
    covariate: str | None = None,
) -> dict:
    """
    Run a two-sample hypothesis test comparing a metric between two groups.

    - continuous metrics -> Welch's t-test (robust to unequal variances)
    - binary metrics -> two-proportion z-test
    - use_cuped=True (continuous only) applies CUPED variance reduction
      using the named covariate column before testing.

    Reported means (`mean_a`, `mean_b`) are always raw, pre-adjustment.
    The effect, p-value, CI, and statistic reflect the adjusted values
    when CUPED is applied.
    """
    # ---- validate CUPED inputs ----
    if use_cuped:
        if metric_type != "continuous":
            raise ValueError("CUPED is only supported for continuous metrics.")
        if covariate is None:
            raise ValueError("use_cuped=True requires a covariate column name.")
        if covariate not in df.columns:
            raise ValueError(f"Covariate column {covariate!r} not in dataframe.")

    # ---- identify groups ----
    groups = sorted(df[group_col].dropna().unique())
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}: {groups}"
    g1, g2 = groups

    # ---- extract paired data (drops rows missing any required column) ----
    cols = [group_col, metric, covariate] if use_cuped else [group_col, metric]
    sub = df[cols].dropna()
    a_raw = sub.loc[sub[group_col] == g1, metric].values.astype(float)
    b_raw = sub.loc[sub[group_col] == g2, metric].values.astype(float)

    # ---- apply CUPED if requested ----
    variance_reduction = None
    if use_cuped:
        a_cov = sub.loc[sub[group_col] == g1, covariate].values.astype(float)
        b_cov = sub.loc[sub[group_col] == g2, covariate].values.astype(float)
        outcome_pooled = np.concatenate([a_raw, b_raw])
        cov_pooled = np.concatenate([a_cov, b_cov])
        adj_pooled, _, variance_reduction = _cuped_adjust(outcome_pooled, cov_pooled)
        n_a = len(a_raw)
        a_test, b_test = adj_pooled[:n_a], adj_pooled[n_a:]
    else:
        a_test, b_test = a_raw, b_raw

    # ---- run the test ----
    if metric_type == "continuous":
        t_stat, p_value = stats.ttest_ind(a_test, b_test, equal_var=False)
        effect = float(b_test.mean() - a_test.mean())
        se = np.sqrt(a_test.var(ddof=1) / len(a_test) + b_test.var(ddof=1) / len(b_test))
        dof = (se ** 4) / (
            (a_test.var(ddof=1) / len(a_test)) ** 2 / (len(a_test) - 1)
            + (b_test.var(ddof=1) / len(b_test)) ** 2 / (len(b_test) - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        ci_low, ci_high = effect - t_crit * se, effect + t_crit * se
        statistic = float(t_stat)

    elif metric_type == "binary":
        p1, p2 = a_test.mean(), b_test.mean()
        n1, n2 = len(a_test), len(b_test)
        p_pool = (a_test.sum() + b_test.sum()) / (n1 + n2)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_stat = (p2 - p1) / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        effect = float(p2 - p1)
        se_ci = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        ci_low, ci_high = effect - 1.96 * se_ci, effect + 1.96 * se_ci
        statistic = float(z_stat)

    else:
        raise ValueError(f"metric_type must be 'continuous' or 'binary', got {metric_type!r}")

    return {
        "group_a": str(g1),
        "group_b": str(g2),
        "n_a": int(len(a_raw)),
        "n_b": int(len(b_raw)),
        "mean_a": float(a_raw.mean()),
        "mean_b": float(b_raw.mean()),
        "absolute_effect": effect,
        "relative_effect": float(effect / a_raw.mean()) if a_raw.mean() != 0 else None,
        "p_value": float(p_value),
        "statistic": statistic,
        "ci_95": [float(ci_low), float(ci_high)],
        "significant": bool(p_value < alpha),
        "cuped_applied": use_cuped,
        "variance_reduction": float(variance_reduction) if variance_reduction is not None else None,
    }