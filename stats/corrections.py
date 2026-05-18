"""Multiple-hypothesis-testing corrections.

When more than one statistical test is run on the same experiment (e.g.,
testing several metrics, or comparing several variants to a control), the
probability of at least one false positive grows with the number of tests:

    P(at least one false positive | all nulls true) = 1 - (1 - alpha)^m

Two correction methods are supported:

- Benjamini-Hochberg ("bh") -- controls the False Discovery Rate (FDR),
  the expected proportion of false positives among rejected nulls. Less
  conservative; the right default for tech experimentation where missing
  a real effect is also costly. Reference: Benjamini & Hochberg (1995).

- Bonferroni ("bonferroni") -- controls the Family-Wise Error Rate
  (FWER), the probability of any false positive across the whole family.
  Very conservative; appropriate for clinical or safety-critical
  decisions where one false positive is unacceptable.

Both methods preserve the original p-values and add adjusted p-values
and a corrected significance flag. The caller chooses which to use.
"""
from statsmodels.stats.multitest import multipletests


_METHOD_MAP = {
    "bh": "fdr_bh",
    "bonferroni": "bonferroni",
}

_METHOD_LABEL = {
    "bh": "Benjamini-Hochberg (FDR control)",
    "bonferroni": "Bonferroni (FWER control)",
}


def multiple_comparisons_correction(
    df=None,  # unused; accepted for router compatibility
    *,
    tests: list[dict],
    method: str = "bh",
    alpha: float = 0.05,
) -> dict:
    """
    Apply multiple-comparisons correction to a set of p-values.

    Parameters
    ----------
    tests  : list of {"metric": str, "p_value": float} dicts -- one per
             statistical test that has been run.
    method : "bh" (default) or "bonferroni".
    alpha  : Significance level / FDR target. Default 0.05.

    Returns a dict with per-test adjusted p-values, a corrected
    significance flag for each, summary counts, and a verdict string
    the LLM should quote.
    """
    # ---- validate inputs ----
    if not isinstance(tests, list) or len(tests) == 0:
        raise ValueError("`tests` must be a non-empty list of dicts.")
    if method not in _METHOD_MAP:
        raise ValueError(
            f"method must be 'bh' or 'bonferroni', got {method!r}"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    for i, t in enumerate(tests):
        if "metric" not in t or "p_value" not in t:
            raise ValueError(
                f"tests[{i}] must have 'metric' and 'p_value' keys, got {t}"
            )
        if not (0 <= t["p_value"] <= 1):
            raise ValueError(
                f"tests[{i}]['p_value'] must be in [0, 1], got {t['p_value']}"
            )

    p_values = [t["p_value"] for t in tests]
    metrics = [t["metric"] for t in tests]
    n_tests = len(tests)
    n_sig_uncorrected = sum(1 for p in p_values if p < alpha)

    # ---- apply correction ----
    reject, p_adjusted, _, _ = multipletests(
        p_values, alpha=alpha, method=_METHOD_MAP[method]
    )
    n_sig_corrected = int(sum(reject))

    results = [
        {
            "metric": metrics[i],
            "p_value": float(p_values[i]),
            "p_value_adjusted": float(p_adjusted[i]),
            "significant": bool(reject[i]),
        }
        for i in range(n_tests)
    ]

    # ---- compose verdict ----
    lost_to_correction = n_sig_uncorrected - n_sig_corrected
    method_label = _METHOD_LABEL[method]

    verdict_lines = [
        f"Applied {method_label} correction across {n_tests} tests at "
        f"alpha={alpha}.",
        f"Significant before correction: {n_sig_uncorrected}/{n_tests}. "
        f"Significant after correction: {n_sig_corrected}/{n_tests}.",
    ]
    if lost_to_correction > 0:
        verdict_lines.append(
            f"{lost_to_correction} metric(s) lost significance after "
            f"correcting for multiple testing -- treat those with caution."
        )
    elif n_sig_corrected == n_sig_uncorrected and n_sig_corrected > 0:
        verdict_lines.append(
            "All originally-significant results survived correction."
        )
    elif n_sig_corrected == 0:
        verdict_lines.append(
            "No metric is significant after correction."
        )

    return {
        "tool": "multiple_comparisons_correction",
        "method": method,
        "alpha": alpha,
        "n_tests": n_tests,
        "n_significant_uncorrected": n_sig_uncorrected,
        "n_significant_corrected": n_sig_corrected,
        "results": results,
        "verdict": " ".join(verdict_lines),
    }
