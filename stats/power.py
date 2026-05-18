"""Power analysis for A/B experiment planning.

Solves the four-way relationship between sample size, effect size, significance
level, and power. Specify any three; the function solves for the fourth.

Continuous metrics use the Welch t-test formulation; binary metrics use the
two-proportion z-test formulation. Both assume equal allocation across arms
(50/50 split), which is the standard industry default.
"""
import math

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


def power_analysis(
    df: pd.DataFrame | None = None,
    *,
    metric_type: str = "continuous",
    solve_for: str = "n",
    # ---- continuous metric inputs ----
    baseline_std: float | None = None,
    baseline_column: str | None = None,
    # ---- binary metric inputs ----
    baseline_rate: float | None = None,
    # ---- planning parameters ----
    mde: float | None = None,
    n_per_arm: int | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Compute one of: required sample size, minimum detectable effect, or power.

    Parameters
    ----------
    metric_type : "continuous" or "binary"
    solve_for   : "n" (sample size), "mde" (minimum detectable effect),
                  or "power"
    baseline_std    : Per-user standard deviation. Required for continuous
                      metrics unless baseline_column is given.
    baseline_column : Column in df to compute std from. Alternative to
                      providing baseline_std directly.
    baseline_rate   : Baseline conversion rate in [0, 1]. Required for
                      binary metrics.
    mde   : Minimum detectable effect. For continuous metrics, in the
            metric's units (e.g. dollars). For binary metrics, in
            absolute percentage points (e.g. 0.02 means a +2pp lift).
    n_per_arm : Sample size per arm.
    alpha     : Significance level. Default 0.05.
    power     : Statistical power (1 - beta). Default 0.80.

    Returns
    -------
    dict with the solved quantity, the inputs used, and a human-readable
    verdict string the LLM can quote.
    """
    # ---- input validation ----
    if solve_for not in {"n", "mde", "power"}:
        raise ValueError(f"solve_for must be 'n', 'mde', or 'power', got {solve_for!r}")
    if metric_type not in {"continuous", "binary"}:
        raise ValueError(f"metric_type must be 'continuous' or 'binary', got {metric_type!r}")

    # ---- compute baseline_std from column if requested (continuous only) ----
    if metric_type == "continuous":
        if baseline_std is None and baseline_column is not None:
            if df is None:
                raise ValueError("baseline_column requires df to be provided.")
            if baseline_column not in df.columns:
                raise ValueError(f"Column {baseline_column!r} not in dataframe.")
            baseline_std = float(df[baseline_column].dropna().std(ddof=1))
        if baseline_std is None and solve_for != "power":
            raise ValueError(
                "Continuous metrics require baseline_std or baseline_column "
                "(unless solving for power, which can use raw MDE/std ratio)."
            )

    if metric_type == "binary":
        if baseline_rate is None:
            raise ValueError("Binary metrics require baseline_rate in [0, 1].")
        if not (0 < baseline_rate < 1):
            raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")

    # ---- branch on solve_for ----
    if metric_type == "continuous":
        analysis = TTestIndPower()

        if solve_for == "n":
            if mde is None:
                raise ValueError("solve_for='n' requires mde.")
            effect_size = mde / baseline_std       # Cohen's d
            n = analysis.solve_power(
                effect_size=effect_size, alpha=alpha, power=power, ratio=1.0,
                alternative="two-sided",
            )
            n_rounded = int(math.ceil(n))
            verdict = (
                f"Need approximately {n_rounded:,} users per arm "
                f"({2 * n_rounded:,} total) to detect an MDE of {mde:g} "
                f"in a continuous metric with std {baseline_std:.3g} "
                f"at alpha={alpha} and {power:.0%} power."
            )
            return {
                "tool": "power_analysis",
                "metric_type": "continuous",
                "solved_for": "n_per_arm",
                "n_per_arm": n_rounded,
                "n_total": 2 * n_rounded,
                "mde": mde,
                "baseline_std": baseline_std,
                "alpha": alpha,
                "power": power,
                "verdict": verdict,
            }

        if solve_for == "mde":
            if n_per_arm is None:
                raise ValueError("solve_for='mde' requires n_per_arm.")
            effect_size = analysis.solve_power(
                effect_size=None, nobs1=n_per_arm, alpha=alpha, power=power,
                ratio=1.0, alternative="two-sided",
            )
            mde_solved = float(effect_size * baseline_std)
            verdict = (
                f"With {n_per_arm:,} users per arm, the smallest effect "
                f"detectable at alpha={alpha} and {power:.0%} power is "
                f"approximately {mde_solved:.4g} (in metric units). "
                f"Effects smaller than this will likely show as inconclusive."
            )
            return {
                "tool": "power_analysis",
                "metric_type": "continuous",
                "solved_for": "mde",
                "mde": mde_solved,
                "n_per_arm": n_per_arm,
                "baseline_std": baseline_std,
                "alpha": alpha,
                "power": power,
                "verdict": verdict,
            }

        # solve_for == "power"
        if n_per_arm is None or mde is None:
            raise ValueError("solve_for='power' requires both n_per_arm and mde.")
        if baseline_std is None:
            raise ValueError("solve_for='power' requires baseline_std for continuous metrics.")
        effect_size = mde / baseline_std
        achieved = analysis.solve_power(
            effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, power=None,
            ratio=1.0, alternative="two-sided",
        )
        verdict = (
            f"With {n_per_arm:,} users per arm and MDE={mde:g} "
            f"(std={baseline_std:.3g}), achieved power is {achieved:.1%} "
            f"at alpha={alpha}. "
            + ("This is adequately powered (>=80%)."
               if achieved >= 0.80
               else "This is underpowered; consider increasing sample size.")
        )
        return {
            "tool": "power_analysis",
            "metric_type": "continuous",
            "solved_for": "power",
            "power": float(achieved),
            "n_per_arm": n_per_arm,
            "mde": mde,
            "baseline_std": baseline_std,
            "alpha": alpha,
            "verdict": verdict,
        }

    # ---- binary metric branch ----
    analysis = NormalIndPower()

    if solve_for == "n":
        if mde is None:
            raise ValueError("solve_for='n' requires mde.")
        p2 = baseline_rate + mde
        if not (0 < p2 < 1):
            raise ValueError(
                f"baseline_rate + mde = {p2} must be in (0, 1). "
                f"Check that mde is in absolute percentage points (e.g. 0.02 not 2)."
            )
        effect_size = proportion_effectsize(baseline_rate, p2)
        n = analysis.solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=1.0,
            alternative="two-sided",
        )
        n_rounded = int(math.ceil(abs(n)))
        verdict = (
            f"Need approximately {n_rounded:,} users per arm "
            f"({2 * n_rounded:,} total) to detect a lift from "
            f"{baseline_rate:.1%} to {p2:.1%} (absolute +{mde:.1%}) "
            f"at alpha={alpha} and {power:.0%} power."
        )
        return {
            "tool": "power_analysis",
            "metric_type": "binary",
            "solved_for": "n_per_arm",
            "n_per_arm": n_rounded,
            "n_total": 2 * n_rounded,
            "mde": mde,
            "baseline_rate": baseline_rate,
            "treatment_rate": p2,
            "alpha": alpha,
            "power": power,
            "verdict": verdict,
        }

    if solve_for == "mde":
        if n_per_arm is None:
            raise ValueError("solve_for='mde' requires n_per_arm.")
        # solve for effect_size in Cohen's h, then invert to absolute pp
        effect_size = analysis.solve_power(
            effect_size=None, nobs1=n_per_arm, alpha=alpha, power=power,
            ratio=1.0, alternative="two-sided",
        )
        # Cohen's h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))
        # invert: p2 = sin(arcsin(sqrt(p1)) + h/2)^2
        p2 = math.sin(math.asin(math.sqrt(baseline_rate)) + effect_size / 2) ** 2
        mde_solved = float(p2 - baseline_rate)
        verdict = (
            f"With {n_per_arm:,} users per arm at baseline rate "
            f"{baseline_rate:.1%}, the smallest absolute lift detectable "
            f"at alpha={alpha} and {power:.0%} power is approximately "
            f"+{mde_solved:.2%} (i.e. {baseline_rate:.1%} -> {p2:.1%})."
        )
        return {
            "tool": "power_analysis",
            "metric_type": "binary",
            "solved_for": "mde",
            "mde": mde_solved,
            "treatment_rate": float(p2),
            "n_per_arm": n_per_arm,
            "baseline_rate": baseline_rate,
            "alpha": alpha,
            "power": power,
            "verdict": verdict,
        }

    # solve_for == "power"
    if n_per_arm is None or mde is None:
        raise ValueError("solve_for='power' requires both n_per_arm and mde.")
    p2 = baseline_rate + mde
    if not (0 < p2 < 1):
        raise ValueError(
            f"baseline_rate + mde = {p2} must be in (0, 1). "
            f"Check that mde is in absolute percentage points."
        )
    effect_size = proportion_effectsize(baseline_rate, p2)
    achieved = analysis.solve_power(
        effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, power=None,
        ratio=1.0, alternative="two-sided",
    )
    verdict = (
        f"With {n_per_arm:,} users per arm and a target lift from "
        f"{baseline_rate:.1%} to {p2:.1%}, achieved power is "
        f"{achieved:.1%} at alpha={alpha}. "
        + ("This is adequately powered (>=80%)."
           if achieved >= 0.80
           else "This is underpowered; consider increasing sample size.")
    )
    return {
        "tool": "power_analysis",
        "metric_type": "binary",
        "solved_for": "power",
        "power": float(achieved),
        "n_per_arm": n_per_arm,
        "mde": mde,
        "baseline_rate": baseline_rate,
        "treatment_rate": p2,
        "alpha": alpha,
        "verdict": verdict,
    }
