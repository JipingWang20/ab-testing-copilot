"""Streamlit UI for the A/B Testing Copilot."""
import pandas as pd
import streamlit as st

from copilot.router import answer_question

st.set_page_config(page_title="A/B Testing Copilot", layout="wide")
st.title("A/B Testing Copilot")
st.caption(
    "Ask questions about your experiment in plain English. "
    "The LLM picks the right statistical test; all numbers come from scipy."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload your own CSV", type="csv")
    use_demo = st.checkbox("Or use the Cookie Cats demo dataset", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_demo:
    df = pd.read_csv("data/cookie_cats.csv")
else:
    st.info("Upload a CSV or check the demo dataset box to get started.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Data preview")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Columns")
    st.write({c: str(df[c].dtype) for c in df.columns})

st.subheader("Ask a question")

examples = [
    "Is the 7-day retention different between the two gate versions? Is it significant?",
    "Does moving the gate from level 30 to 40 affect average game rounds played?",
    "Is day-1 retention significantly impacted by the gate change?",
]
example = st.selectbox("Try an example:", [""] + examples)
question = st.text_input(
    "Your question:",
    value=example,
    placeholder="e.g. Is the conversion lift statistically significant?",
)

if st.button("Analyze", type="primary", disabled=not question):
    with st.spinner("The copilot is thinking and running tests..."):
        result = answer_question(df, question)

    st.subheader("Answer")
    st.markdown(result["answer"])

    with st.expander(f"Tool calls ({len(result['tool_calls'])}) - full transparency"):
        for i, tc in enumerate(result["tool_calls"], 1):
            st.markdown(f"**{i}. `{tc['tool']}`**")
            st.json({"args": tc["args"], "result": tc["result"]})
