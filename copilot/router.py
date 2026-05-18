"""LLM router: natural language question -> statistical tool call -> answer."""
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from stats.tests import two_sample_test
from stats.diagnostics import srm_check, cuped_check
from stats.power import power_analysis


load_dotenv()  # <-- must come BEFORE Anthropic() so the key is in env
client = Anthropic()
MODEL = "claude-sonnet-4-5"

# ============ Tool schemas the LLM can see ============
TOOLS = [
    {
        "name": "two_sample_test",
        "description": (
            "Test whether a metric differs significantly between two groups in "
            "an A/B experiment. Automatically picks Welch's t-test for continuous "
            "metrics and two-proportion z-test for binary metrics. "
            "Optionally applies CUPED variance reduction (continuous metrics only) "
            "when a pre-experiment covariate is supplied via use_cuped=True. "
            "Returns effect size, p-value, 95% confidence interval, significance "
            "flag, and -- when CUPED is applied -- the achieved variance reduction. "
            "Use this for ANALYSIS of completed experiments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Column name of the metric to test."},
                "group_col": {"type": "string", "description": "Column labelling the two groups."},
                "metric_type": {
                    "type": "string",
                    "enum": ["continuous", "binary"],
                    "description": (
                        "'binary' for 0/1 or True/False outcomes like retention or "
                        "conversion. 'continuous' for counts or real-valued metrics."
                    ),
                },
                "alpha": {"type": "number", "description": "Significance level. Default 0.05."},
                "use_cuped": {
                    "type": "boolean",
                    "description": (
                        "If true, apply CUPED variance reduction using the named "
                        "covariate before testing. Only valid for continuous metrics. "
                        "Set this based on the verdict returned by cuped_check."
                    ),
                },
                "covariate": {
                    "type": "string",
                    "description": "Column name of the pre-experiment covariate for CUPED.",
                },
            },
            "required": ["metric", "group_col", "metric_type"],
        },
    },
    {
        "name": "srm_check",
        "description": (
            "Check for Sample Ratio Mismatch (SRM). Runs a chi-square goodness-of-fit "
            "test on group sizes against an expected split. A significant result means "
            "randomization is likely broken -- downstream tests should NOT be trusted. "
            "ALWAYS call this BEFORE interpreting any two_sample_test result. "
            "Do NOT call for planning questions about future experiments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Column labelling the groups."},
                "expected_split": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Expected proportions, e.g. [0.5, 0.5]. Omit for equal split.",
                },
            },
            "required": ["group_col"],
        },
    },
    {
        "name": "cuped_check",
        "description": (
            "Pre-flight check for CUPED variance reduction. Verifies (1) the covariate "
            "is balanced across arms (no leakage), and (2) it is correlated enough with "
            "the outcome. Returns status 'APPLY', 'SKIP', or 'BLOCKED'. Call AFTER "
            "srm_check passes, and ONLY for continuous metrics with a plausible "
            "pre-experiment covariate. Do NOT call for planning questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Outcome metric column."},
                "group_col": {"type": "string", "description": "Group column."},
                "covariate": {
                    "type": "string",
                    "description": "Candidate pre-experiment covariate column.",
                },
            },
            "required": ["metric", "group_col", "covariate"],
        },
    },
    {
        "name": "power_analysis",
        "description": (
            "Power analysis for experiment PLANNING (before the experiment runs). "
            "Solves the four-way relationship between sample size per arm (n), "
            "minimum detectable effect (mde), significance level (alpha), and power. "
            "Specify three of the four; the function solves for the remaining one via "
            "solve_for. Use this when the user asks how many users they need, what "
            "effect they could detect with a given sample, or whether a planned "
            "experiment is adequately powered. Do NOT use this for analyzing experiments "
            "that have already completed -- use two_sample_test for that. For continuous "
            "metrics, pass either baseline_std (a number) or baseline_column (a column "
            "name to compute std from). For binary metrics, pass baseline_rate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "enum": ["continuous", "binary"],
                    "description": (
                        "'continuous' for real-valued metrics like revenue. "
                        "'binary' for 0/1 outcomes like conversion or retention."
                    ),
                },
                "solve_for": {
                    "type": "string",
                    "enum": ["n", "mde", "power"],
                    "description": (
                        "'n' = solve for required sample size per arm. "
                        "'mde' = solve for smallest detectable effect given fixed n. "
                        "'power' = solve for achieved power given fixed n and mde."
                    ),
                },
                "baseline_std": {
                    "type": "number",
                    "description": "Per-user std for continuous metrics. Use this OR baseline_column.",
                },
                "baseline_column": {
                    "type": "string",
                    "description": (
                        "Column name to compute std from (e.g. a pre-period revenue column). "
                        "Alternative to baseline_std."
                    ),
                },
                "baseline_rate": {
                    "type": "number",
                    "description": "Baseline conversion rate in (0, 1) for binary metrics.",
                },
                "mde": {
                    "type": "number",
                    "description": (
                        "Minimum detectable effect. Continuous: in metric units (e.g. 1.0 "
                        "means $1 lift). Binary: absolute percentage points (e.g. 0.02 means +2pp)."
                    ),
                },
                "n_per_arm": {"type": "integer", "description": "Sample size per arm."},
                "alpha": {"type": "number", "description": "Significance level. Default 0.05."},
                "power": {"type": "number", "description": "Target power. Default 0.80."},
            },
            "required": ["metric_type", "solve_for"],
        },
    },
]

TOOL_REGISTRY = {
    "two_sample_test": two_sample_test,
    "srm_check": srm_check,
    "cuped_check": cuped_check,
    "power_analysis": power_analysis,
}

SYSTEM_PROMPT = """You are a statistical copilot for A/B testing.

You have access to tools that run rigorous statistical computations. You
NEVER compute p-values, confidence intervals, sample sizes, or effect
sizes yourself -- you ALWAYS call the appropriate tool.

First, identify whether the user is asking about:

(A) ANALYSIS of an experiment that already happened (e.g. "did the
    treatment lift revenue?", "is this difference significant?"). Use
    the ANALYSIS WORKFLOW below.

(B) PLANNING a future experiment (e.g. "how many users do I need?",
    "what effect could I detect with 5000 users per arm?", "is my
    planned experiment well-powered?"). Use the PLANNING WORKFLOW below.

==================== ANALYSIS WORKFLOW ====================

(1) Randomization check. Before interpreting any two_sample_test result,
you MUST first call srm_check. Quote the verdict faithfully.
- If "PASSED", briefly mention it and move on. Do not add caveats.
- If "FAILED", flag prominently and recommend investigating.

(2) CUPED pre-flight (continuous metrics only). After srm_check passes,
if the dataset contains a column whose name suggests pre-experiment
capture (starts with "pre_", "baseline_", or contains "_pre"), AND the
primary metric is continuous, you MUST call cuped_check with that column
as the covariate. Honor the verdict:
- "APPLY"   -> two_sample_test with use_cuped=True and same covariate.
- "SKIP"    -> two_sample_test with use_cuped=False.
- "BLOCKED" -> surface verdict prominently, do NOT use CUPED.

For binary metrics, skip cuped_check entirely.

(3) Primary test. Call two_sample_test with use_cuped per the verdict.
The returned p-value, CI, and effect already reflect any CUPED
adjustment -- use them directly.

(4) Summary. Plain-English PM-style summary. Always include:
- effect direction and size (pp for binary metrics)
- significance, with p-value and CI cited
- clear recommendation (ship / don't ship / inconclusive)
- one sentence on CUPED if applied (and the variance reduction)

==================== PLANNING WORKFLOW ====================

Call power_analysis. Do NOT call srm_check or cuped_check for planning
questions -- those are for completed experiments only.

Picking inputs:
- metric_type: infer from context ("revenue" -> continuous;
  "conversion" / "retention" -> binary).
- solve_for: "n" for "how many users?", "mde" for "what effect could
  I detect with N users?", "power" for "is this plan well-powered?".
- For continuous: pass baseline_column if the dataset has a relevant
  pre-period or historical column; otherwise pass baseline_std directly.
  If neither is available, ask the user for a baseline_std estimate.
- For binary: pass baseline_rate. If the dataset has the relevant
  binary column, you may compute and mention the rate; otherwise ask.
- mde: must come from the user. If unspecified, ask: "What's the
  smallest effect that would matter to the business?"

Quote the verdict faithfully. In your summary:
- State the answer clearly (e.g. "You need ~5,700 users per arm").
- Briefly explain what it means for the user's plan.
- If n is impractically large, suggest alternatives: longer duration,
  larger MDE, or CUPED if pre-period data is available.

==================== HARD RULES ====================
- Never compute or re-derive any statistic from raw inputs.
- Never override a tool's verdict by reasoning independently.
- If tools conflict, surface the conflict explicitly.

Dataset columns: {columns}
Column dtypes: {dtypes}
"""

def answer_question(df: pd.DataFrame, question: str, max_hops: int = 6) -> dict:
    """Run a multi-turn function-calling loop until Claude gives a final answer."""
    system = SYSTEM_PROMPT.format(
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
    )

    messages = [{"role": "user", "content": question}]
    tool_calls = []

    for _ in range(max_hops):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"answer": text, "tool_calls": tool_calls}

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_REGISTRY[block.name]
                result = fn(df=df, **block.input)
                tool_calls.append({
                    "tool": block.name,
                    "args": block.input,
                    "result": result,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "user", "content": tool_results})

    # ---- hop budget exhausted ----
    return {
        "answer": (
            f"Reached the maximum of {max_hops} tool-calling hops without a "
            "final answer. Inspect tool_calls for what was attempted."
        ),
        "tool_calls": tool_calls,
    }