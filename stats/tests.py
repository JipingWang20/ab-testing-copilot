"""Two-sample statistical tests for A/B experiments."""
import numpy as np
import pandas as pd
from scipy import stats


def two_sample_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    metric_type: str = "continuous",
    alpha: float = 0.05,
) -> dict:
    """
    Run a two-sample hypothesis test comparing a metric between two groups.

    - continuous metrics -> Welch's t-test (robust to unequal variances)
    - binary metrics -> two-proportion z-test

    Returns a dict with effect size, p-value, 95% CI, and a significance flag.
    """
    groups = sorted(df[group_col].dropna().unique())
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}: {groups}"
    g1, g2 = groups

    x = df.loc[df[group_col] == g1, metric].dropna().values.astype(float)
    y = df.loc[df[group_col] == g2, metric].dropna().values.astype(float)

    if metric_type == "continuous":
        t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
        effect = float(y.mean() - x.mean())
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        dof = (se ** 4) / (
            (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)
            + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        ci_low, ci_high = effect - t_crit * se, effect + t_crit * se
        statistic = float(t_stat)

    elif metric_type == "binary":
        p1, p2 = x.mean(), y.mean()
        n1, n2 = len(x), len(y)
        p_pool = (x.sum() + y.sum()) / (n1 + n2)
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
        "n_a": int(len(x)),
        "n_b": int(len(y)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "absolute_effect": effect,
        "relative_effect": float(effect / x.mean()) if x.mean() != 0 else None,
        "p_value": float(p_value),
        "statistic": statistic,
        "ci_95": [float(ci_low), float(ci_high)],
        "significant": bool(p_value < alpha),
    }
