"""LLM router: natural language question -> statistical tool call -> answer."""
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from stats.tests import two_sample_test
from stats.diagnostics import srm_check, cuped_check


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
            "flag, and -- when CUPED is applied -- the achieved variance reduction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Column name of the metric to test (e.g. 'retention_7').",
                },
                "group_col": {
                    "type": "string",
                    "description": "Column name that labels the two groups (e.g. 'version').",
                },
                "metric_type": {
                    "type": "string",
                    "enum": ["continuous", "binary"],
                    "description": (
                        "'binary' for 0/1 or True/False outcomes like retention or "
                        "conversion. 'continuous' for counts or real-valued metrics "
                        "like revenue or gamerounds."
                    ),
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level. Default 0.05.",
                },
                "use_cuped": {
                    "type": "boolean",
                    "description": (
                        "If true, apply CUPED variance reduction using the named "
                        "covariate before testing. Only valid for continuous metrics. "
                        "Set this based on the verdict returned by cuped_check -- do "
                        "NOT decide on your own."
                    ),
                },
                "covariate": {
                    "type": "string",
                    "description": (
                        "Column name of the pre-experiment covariate to use for "
                        "CUPED. Required when use_cuped is true."
                    ),
                },
            },
            "required": ["metric", "group_col", "metric_type"],
        },
    },
    {
        "name": "srm_check",
        "description": (
            "Check for Sample Ratio Mismatch (SRM) in an A/B experiment. "
            "Runs a chi-square goodness-of-fit test on group sizes against "
            "an expected split (default 50/50). A significant result means "
            "randomization is likely broken -- downstream test results "
            "should NOT be trusted until the root cause is found. "
            "ALWAYS call this BEFORE interpreting any two_sample_test result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column labelling the experiment groups.",
                },
                "expected_split": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Expected group proportions, e.g. [0.5, 0.5] for a "
                        "balanced experiment. Omit for equal split."
                    ),
                },
            },
            "required": ["group_col"],
        },
    },
    {
        "name": "cuped_check",
        "description": (
            "Pre-flight check for CUPED variance reduction. Decides whether a "
            "candidate pre-experiment covariate is suitable for CUPED by "
            "verifying (1) the covariate is balanced across arms (no leakage / "
            "post-treatment contamination), and (2) it is correlated enough with "
            "the outcome to be worth using. Returns one of three statuses: "
            "'APPLY' (use CUPED), 'SKIP' (covariate too weak), or 'BLOCKED' "
            "(covariate fails balance check). Call AFTER srm_check passes, and "
            "ONLY for continuous metrics with a plausible pre-experiment covariate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Column name of the outcome metric.",
                },
                "group_col": {
                    "type": "string",
                    "description": "Column labelling the experiment groups.",
                },
                "covariate": {
                    "type": "string",
                    "description": (
                        "Column name of the candidate pre-experiment covariate "
                        "(e.g. 'pre_revenue', 'baseline_value')."
                    ),
                },
            },
            "required": ["metric", "group_col", "covariate"],
        },
    },
]

TOOL_REGISTRY = {
    "two_sample_test": two_sample_test,
    "srm_check": srm_check,
    "cuped_check": cuped_check,
}

SYSTEM_PROMPT = """You are a statistical copilot for A/B testing analysis.

You have access to tools that run rigorous statistical tests. You NEVER
compute p-values, confidence intervals, or effect sizes yourself -- you
ALWAYS call the appropriate tool.

WORKFLOW:

(1) Randomization check. Before interpreting any two_sample_test result,
you MUST first call srm_check to verify that randomization is not broken.
The srm_check tool returns a status ("PASSED" or "FAILED") and a verdict
string. Quote the verdict faithfully.
- If status is "PASSED", briefly mention that the randomization check
  passed and move on. Do not add caveats.
- If status is "FAILED", flag the issue prominently and recommend
  investigating before trusting the primary test.

(2) CUPED pre-flight (continuous metrics only). After srm_check passes,
if the dataset contains a column whose name suggests it was captured
BEFORE the experiment started (e.g., starts with "pre_", "baseline_",
or contains "_pre"), AND the primary metric is continuous, you MUST call
cuped_check with that column as the covariate. The cuped_check tool
returns a status:
- "APPLY"   : the covariate is suitable. Call two_sample_test with
              use_cuped=True and the same covariate.
- "SKIP"    : the covariate is too weakly correlated with the outcome.
              Call two_sample_test normally (use_cuped=False).
- "BLOCKED" : the covariate fails the balance check (likely leakage or
              broken randomization on that covariate). Surface the
              verdict prominently and do NOT use CUPED.
Quote the verdict faithfully. Do not second-guess it.

For binary metrics, skip cuped_check entirely -- CUPED is not supported
for proportions in this tool.

(3) Primary test. Call two_sample_test with use_cuped set according to
the cuped_check verdict above. The returned p-value, CI, and effect size
already reflect the CUPED adjustment when use_cuped=True -- use them
directly. Never recompute or sanity-check them.

(4) Summary. When you get the two_sample_test result back, write a short
plain-English summary that a non-technical product manager would
understand. Always include:
- the direction and size of the effect (in percentage points for binary
  metrics)
- whether it is statistically significant, and why (cite the p-value
  and CI)
- a clear recommendation (ship / don't ship / inconclusive)
- if CUPED was applied, one sentence noting it and the reported
  variance reduction (e.g., "Variance reduced 34% via CUPED using
  pre-period revenue."). Do not explain the math unless asked.
- if CUPED was considered but skipped or blocked, one brief sentence
  saying so.

HARD RULES:
- Never compute, estimate, or re-derive any statistic from raw inputs.
- Never override a tool's verdict by reasoning independently.
- If two tools return conflicting signals, surface the conflict
  explicitly rather than silently picking one.

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
            "final answer. This usually means the question requires more "
            "diagnostic steps than the budget allows, or the model is "
            "looping. Inspect tool_calls for what was attempted."
        ),
        "tool_calls": tool_calls,
    }