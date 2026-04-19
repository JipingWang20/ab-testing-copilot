"""LLM router: natural language question -> statistical tool call -> answer."""
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from stats.tests import two_sample_test

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
            "metrics and two-proportion z-test for binary metrics. Returns effect "
            "size, p-value, 95% confidence interval, and significance flag."
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
            },
            "required": ["metric", "group_col", "metric_type"],
        },
    }
]

TOOL_REGISTRY = {
    "two_sample_test": two_sample_test,
}

SYSTEM_PROMPT = """You are a statistical copilot for A/B testing analysis.

You have access to tools that run rigorous statistical tests. You NEVER
compute p-values, confidence intervals, or effect sizes yourself -- you
ALWAYS call the appropriate tool.

When you get the tool result back, write a short plain-English summary that
a non-technical product manager would understand. Always include:
- the direction and size of the effect (in percentage points for binary metrics)
- whether it is statistically significant, and why (cite the p-value and CI)
- a clear recommendation (ship / don't ship / inconclusive)

Dataset columns: {columns}
Column dtypes: {dtypes}
"""


def answer_question(df: pd.DataFrame, question: str, max_hops: int = 5) -> dict:
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

    return {"answer": "Hit max tool-call limit.", "tool_calls": tool_calls}
