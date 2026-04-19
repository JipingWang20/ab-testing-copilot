# A/B Testing Copilot

> Ask A/B testing questions in plain English. Get statistically rigorous answers.

An LLM-powered copilot for A/B test analysis. Upload an experiment CSV, ask a question in English, and get back a proper statistical answer — effect size, p-value, 95% confidence interval, and a clear ship / don't-ship recommendation.

## What makes this different

Most LLM demos ask the language model to do statistics. **This one doesn't.** The LLM's only jobs are picking the right test and explaining the output in plain English. Every number comes from `scipy`.

The separation is deliberate. LLMs are unreliable at arithmetic — asking Claude directly for a p-value gets you a confident-sounding wrong answer. The copilot architecture avoids this entirely by using **function calling**: Claude reads the question, emits a structured request like `two_sample_test(metric="retention_7", metric_type="binary")`, and lets Python do the math. The final natural-language summary is wrapped around real numbers, not hallucinated ones.

## Example

**Question:** *"Is the 7-day retention different between the two gate versions?"*

**Answer:**
> Yes, there is a significant difference in 7-day retention between the two versions. gate_40 has lower retention than gate_30 (18.2% vs 19.0%), a decrease of 0.82 percentage points (p = 0.0016, 95% CI [-1.33%, -0.31%]).
>
> **Recommendation: Do NOT ship gate_40.**

All statistics computed via `scipy.stats` (two-proportion z-test). The LLM only picked the test and wrote the summary.

## Architecture

Question (English) → Claude (function calling, picks the right tool) → scipy / statsmodels (computes p-value, CI, effect size) → Claude (synthesis, writes plain-English recommendation) → Streamlit UI

## Current capabilities

- **Two-sample hypothesis tests**: Welch's t-test for continuous metrics, two-proportion z-test for binary metrics
- **Automatic metric-type routing**: the LLM infers whether a column is continuous or binary from its dtype and semantic context
- **Transparent tool calls**: every tool call is logged and shown to the user so results are auditable
- **Ships with a demo dataset**: Cookie Cats (90k mobile game players, gate_30 vs gate_40)

## Roadmap

- [ ] CUPED variance reduction when pre-experiment covariates are available
- [ ] Sample ratio mismatch (SRM) check as a pre-flight diagnostic
- [ ] Segment-level analysis with Simpson's paradox detection
- [ ] Benjamini-Hochberg correction for multiple-segment testing
- [ ] Statistical power and minimum detectable effect (MDE) estimation
- [ ] Unit tests against synthetic data with known ground truth

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/ab-testing-copilot
cd ab-testing-copilot

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

streamlit run app.py
```

Then tick "Use the Cookie Cats demo dataset" in the sidebar and try one of the example questions.

## Project structure

- `app.py` — Streamlit UI
- `copilot/router.py` — LLM function-calling loop
- `stats/tests.py` — scipy-based hypothesis tests
- `data/cookie_cats.csv` — demo dataset
- `requirements.txt`
- `README.md`

## Limitations

- Supports two-arm experiments only (no A/B/n yet)
- Relies on user-supplied column labels; does not infer schema
- Intent routing is probabilistic — the tool-call log exists so mistakes are visible rather than hidden
- No correction for peeking or sequential testing

## About

Built by Jiping Wang, a PhD candidate in Statistics at Florida State University, as a portfolio project demonstrating statistically rigorous use of LLMs for data analysis. Most LLM demos are impressive; few are statistically defensible. This one tries to be both.