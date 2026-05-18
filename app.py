"""Streamlit UI for the A/B Testing Copilot."""
import numpy as np
import pandas as pd
import streamlit as st

from copilot.router import answer_question

st.set_page_config(page_title="A/B Testing Copilot", layout="wide")
st.title("A/B Testing Copilot")
st.caption(
    "Ask questions about your experiment in plain English. "
    "The LLM picks the right statistical test; all numbers come from scipy. "
    "Pre-flight diagnostics run automatically: SRM check on every experiment, "
    "CUPED variance reduction when a pre-period covariate is available."
)


# ============================================================
# Synthetic CUPED demo dataset (generated deterministically)
# ============================================================
@st.cache_data
def make_cuped_demo(n: int = 5000) -> pd.DataFrame:
    """Simulated revenue experiment with a usable pre-period covariate.

    y = 0.6 * pre_revenue + 2 * I(treatment) + N(0, 15).
    Pre-period revenue explains ~40% of outcome variance, so CUPED should
    apply and meaningfully tighten the confidence interval.
    """
    rng = np.random.default_rng(42)
    group = rng.choice(["control", "treatment"], size=n)
    pre_revenue = rng.normal(100, 20, n)
    revenue = (
        0.6 * pre_revenue
        + np.where(group == "treatment", 2.0, 0.0)
        + rng.normal(0, 15, n)
    )
    return pd.DataFrame({
        "group": group,
        "revenue": revenue.round(2),
        "pre_revenue": pre_revenue.round(2),
    })


# ============================================================
# Sidebar: data source
# ============================================================
with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Choose a dataset:",
        ["Cookie Cats (SRM demo)", "Revenue (CUPED demo)", "Upload your own"],
        index=0,
    )
    uploaded = None
    if source == "Upload your own":
        uploaded = st.file_uploader("Upload a CSV", type="csv")

    st.divider()
    st.caption(
        "**Cookie Cats** — classic mobile-game A/B test on retention.\n\n"
        "**Revenue** — synthetic dataset with a pre-period covariate, "
        "designed to exercise CUPED variance reduction."
    )


# ============================================================
# Load dataframe based on source
# ============================================================
if source == "Cookie Cats (SRM demo)":
    df = pd.read_csv("data/cookie_cats.csv")
    dataset_examples = [
        "Is the 7-day retention different between the two gate versions?",
        "Does moving the gate from level 30 to 40 affect average game rounds played?",
        "Is day-1 retention significantly impacted by the gate change?",
    ]
elif source == "Revenue (CUPED demo)":
    df = make_cuped_demo()
    dataset_examples = [
        "Did the treatment lift revenue?",
        "Is the revenue difference between control and treatment statistically significant?",
        "Should we ship the treatment based on revenue?",
    ]
elif uploaded is not None:
    df = pd.read_csv(uploaded)
    dataset_examples = []
else:
    st.info("Choose a demo or upload a CSV to get started.")
    st.stop()


# ============================================================
# Data preview
# ============================================================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data preview")
    st.dataframe(df.head(10), width="stretch")
with col2:
    st.subheader("Columns")
    st.write({c: str(df[c].dtype) for c in df.columns})


# ============================================================
# Question input
# ============================================================
st.subheader("Ask a question")
example = st.selectbox("Try an example:", [""] + dataset_examples)
question = st.text_input(
    "Your question:",
    value=example,
    placeholder="e.g. Is the conversion lift statistically significant?",
)


# ============================================================
# Run analysis
# ============================================================
STATUS_STYLE = {
    "PASSED":  ("✅", "Randomization OK"),
    "FAILED":  ("❌", "SRM detected"),
    "APPLY":   ("✅", "CUPED applied"),
    "SKIP":    ("➖", "CUPED skipped (weak covariate)"),
    "BLOCKED": ("⚠️", "CUPED blocked (balance check failed)"),
}


def find_call(tool_calls, tool_name):
    return next((tc for tc in tool_calls if tc["tool"] == tool_name), None)


if st.button("Analyze", type="primary", disabled=not question):
    with st.spinner("The copilot is thinking and running tests..."):
        result = answer_question(df, question)

    # ---- Diagnostics panel ----
    srm = find_call(result["tool_calls"], "srm_check")
    cuped = find_call(result["tool_calls"], "cuped_check")
    test = find_call(result["tool_calls"], "two_sample_test")

    if srm or cuped or (test and test["result"].get("cuped_applied")):
        st.subheader("Diagnostics")
        cols = st.columns(3)

        if srm:
            icon, label = STATUS_STYLE.get(
                srm["result"]["status"], ("•", srm["result"]["status"])
            )
            cols[0].metric("SRM Check", f"{icon} {srm['result']['status']}", label)

        if cuped:
            icon, label = STATUS_STYLE.get(
                cuped["result"]["status"], ("•", cuped["result"]["status"])
            )
            cols[1].metric(
                "CUPED Pre-flight",
                f"{icon} {cuped['result']['status']}",
                label,
            )

        if test and test["result"].get("cuped_applied"):
            vr = test["result"].get("variance_reduction") or 0.0
            cols[2].metric(
                "Variance Reduction",
                f"{vr:.0%}",
                "via CUPED adjustment",
            )

    # ---- Main answer ----
    st.subheader("Answer")
    st.markdown(result["answer"].replace("$", "\\$"))
    
    # ---- Full trace ----
    with st.expander(
        f"Tool calls ({len(result['tool_calls'])}) — full transparency"
    ):
        for i, tc in enumerate(result["tool_calls"], 1):
            st.markdown(f"**{i}. `{tc['tool']}`**")
            st.json({"args": tc["args"], "result": tc["result"]})