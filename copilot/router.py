"""LLM router: natural language question -> statistical tool call -> answer."""
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from stats.tests import two_sample_test
from stats.diagnostics import srm_check, cuped_check
from stats.power import power_analysis
from stats.corrections import multiple_comparisons_correction


load_dotenv()  # <-- must come BEFORE Anthropic() so the key is in env
client = Anthropic()
MODEL = "claude-sonnet-4-5"

# ============ Tool schemas the LLM can see ============
TOOLS = [
    {
        "name": "two_sample_test",
        "description": (
            "Test whether a metric differs significantly between two groups in "
            "an A/B experiment. Welch's t-test for continuous metrics; two-"
            "proportion z-test for binary metrics. Optionally applies CUPED "
            "variance reduction (continuous only) when use_cuped=True. Use for "
            "ANALYSIS of completed experiments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Metric column."},
                "group_col": {"type": "string", "description": "Group column."},
                "metric_type": {
                    "type": "string",
                    "enum": ["continuous", "binary"],
                    "description": "'binary' for 0/1 outcomes; 'continuous' for real-valued.",
                },
                "alpha": {"type": "number", "description": "Significance level. Default 0.05."},
                "use_cuped": {
                    "type": "boolean",
                    "description": "Apply CUPED. Set per the cuped_check verdict.",
                },
                "covariate": {
                    "type": "string",
                    "description": "Pre-experiment covariate column for CUPED.",
                },
            },
            "required": ["metric", "group_col", "metric_type"],
        },
    },
    {
        "name": "srm_check",
        "description": (
            "Sample Ratio Mismatch check. Chi-square on group sizes vs expected "
            "split. ALWAYS call BEFORE interpreting any two_sample_test result. "
            "Do NOT call for planning questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Group column."},
                "expected_split": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Expected proportions. Omit for equal split.",
                },
            },
            "required": ["group_col"],
        },
    },
    {
        "name": "cuped_check",
        "description": (
            "CUPED pre-flight. Returns APPLY / SKIP / BLOCKED. Call AFTER "
            "srm_check passes, ONLY for continuous metrics with a plausible "
            "pre-experiment covariate. Do NOT call for planning questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Outcome metric."},
                "group_col": {"type": "string", "description": "Group column."},
                "covariate": {"type": "string", "description": "Candidate covariate."},
            },
            "required": ["metric", "group_col", "covariate"],
        },
    },
    {
        "name": "power_analysis",
        "description": (
            "Power analysis for experiment PLANNING. Solves the n / mde / power / "
            "alpha relationship -- specify three, solve for the fourth via "
            "solve_for. Use when the user asks how many users they need, what "
            "effect they could detect with N users, or whether a plan is "
            "well-powered. Do NOT use for analyzing completed experiments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "enum": ["continuous", "binary"],
                    "description": "'continuous' or 'binary'.",
                },
                "solve_for": {
                    "type": "string",
                    "enum": ["n", "mde", "power"],
                    "description": "Which quantity to solve for.",
                },
                "baseline_std": {"type": "number", "description": "Per-user std (continuous)."},
                "baseline_column": {
                    "type": "string",
                    "description": "Column to compute std from. Alternative to baseline_std.",
                },
                "baseline_rate": {"type": "number", "description": "Conversion rate (binary)."},
                "mde": {"type": "number", "description": "Minimum detectable effect."},
                "n_per_arm": {"type": "integer", "description": "Sample size per arm."},
                "alpha": {"type": "number", "description": "Significance level. Default 0.05."},
                "power": {"type": "number", "description": "Target power. Default 0.80."},
            },
            "required": ["metric_type", "solve_for"],
        },
    },
    {
        "name": "multiple_comparisons_correction",
        "description": (
            "Adjust for multiple-hypothesis testing. When more than one "
            "two_sample_test has been run on the same experiment (multiple "
            "metrics, multiple variants, or both), the chance of at least one "
            "false positive grows with the number of tests. Call this AFTER "
            "all two_sample_test calls in an analysis, BEFORE writing the "
            "final summary. Default method 'bh' (Benjamini-Hochberg, controls "
            "false discovery rate) is right for tech experimentation; "
            "'bonferroni' is more conservative (controls family-wise error "
            "rate) and used for clinical or safety-critical decisions. The "
            "final summary should report ADJUSTED significance, not raw."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "p_value": {"type": "number"},
                        },
                        "required": ["metric", "p_value"],
                    },
                    "description": (
                        "List of {metric, p_value} pairs from all "
                        "two_sample_test calls in this analysis."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["bh", "bonferroni"],
                    "description": "Correction method. Default 'bh'.",
                },
                "alpha": {
                    "type": "number",
                    "description": "FDR target or family-wise alpha. Default 0.05.",
                },
            },
            "required": ["tests"],
        },
    },
]

TOOL_REGISTRY = {
    "two_sample_test": two_sample_test,
    "srm_check": srm_check,
    "cuped_check": cuped_check,
    "power_analysis": power_analysis,
    "multiple_comparisons_correction": multiple_comparisons_correction,
}

SYSTEM_PROMPT = """You are a statistical copilot for A/B testing.

You have access to tools that run rigorous statistical computations. You
NEVER compute p-values, confidence intervals, sample sizes, or effect
sizes yourself -- you ALWAYS call the appropriate tool.

First, identify whether the user is asking about:

(A) ANALYSIS of an experiment that already happened (e.g. "did the
    treatment lift revenue?", "is this difference significant?", "did
    retention and revenue both improve?"). Use the ANALYSIS WORKFLOW.

(B) PLANNING a future experiment (e.g. "how many users do I need?",
    "what effect could I detect with 5000 users per arm?"). Use the
    PLANNING WORKFLOW.

==================== ANALYSIS WORKFLOW ====================

(1) Randomization check. Before interpreting any two_sample_test result,
you MUST first call srm_check. Quote the verdict faithfully.
- "PASSED": mention briefly and move on. No caveats.
- "FAILED": flag prominently and recommend investigating.

(2) CUPED pre-flight (continuous metrics only). After srm_check passes,
if the dataset contains a column whose name suggests pre-experiment
capture (starts with "pre_", "baseline_", or contains "_pre"), AND the
metric is continuous, you MUST call cuped_check. Honor the verdict:
- APPLY   -> two_sample_test with use_cuped=True and the covariate.
- SKIP    -> two_sample_test with use_cuped=False.
- BLOCKED -> surface verdict prominently, do NOT use CUPED.
Skip cuped_check entirely for binary metrics.

(3) Primary test(s). Call two_sample_test for each metric the user
asked about. If the question mentions multiple metrics ("did retention
AND revenue improve?"), or compares multiple variants, run two_sample_test
once per metric/variant.

(4) Multiple-comparisons correction (REQUIRED if more than one
two_sample_test was called in this analysis). After ALL two_sample_test
calls, call multiple_comparisons_correction with the list of
{{metric, p_value}} pairs. Use method='bh' by default (FDR control,
appropriate for tech). Use 'bonferroni' only if the user explicitly
asks for family-wise error control, or the context is clinical/safety
critical. In the final summary, report ADJUSTED significance, not raw.
If a metric loses significance under correction, say so explicitly.

(5) Summary. Plain-English PM-style summary. Always include:
- effect direction and size (pp for binary metrics)
- significance (after correction, if applicable), citing p-value/p_adj and CI
- clear recommendation (ship / don't ship / inconclusive)
- one sentence on CUPED if applied (and the variance reduction)
- one sentence on correction if applied (which method, what it changed)

==================== PLANNING WORKFLOW ====================

Call power_analysis. Do NOT call srm_check, cuped_check, or
multiple_comparisons_correction for planning questions -- those are for
completed experiments only.

Picking inputs:
- metric_type: infer from context.
- solve_for: "n" for sample size, "mde" for detectable effect, "power"
  for adequacy of a plan.
- For continuous: pass baseline_column if a relevant pre-period/historical
  column exists; otherwise pass baseline_std directly; otherwise ask.
- For binary: pass baseline_rate.
- mde: must come from the user. If unspecified, ask: "What's the smallest
  effect that would matter to the business?"

Quote the verdict faithfully. State the answer clearly and briefly
explain what it means for the plan. If n is impractically large, suggest
alternatives: longer duration, larger MDE, or CUPED if pre-period data
is available.

==================== HARD RULES ====================
- Never compute or re-derive any statistic from raw inputs.
- Never override a tool's verdict by reasoning independently.
- If tools conflict, surface the conflict explicitly.

Dataset columns: {columns}
Column dtypes: {dtypes}
"""

def answer_question(df: pd.DataFrame, question: str, max_hops: int = 8) -> dict:
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