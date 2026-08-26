"""
Credit Default Prediction & Portfolio Risk Dashboard
-----------------------------------------------------
Streamlit app: portfolio-level risk KPIs, risk-band and segment breakdowns,
and a drill-down into the highest-risk accounts.

Run:
    python src/download_data.py
    python src/train_model.py
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from risk_metrics import (  # noqa: E402
    load_scored_portfolio,
    portfolio_summary,
    risk_band_breakdown,
    segment_breakdown,
    top_risk_accounts,
)
from scoring import score_raw_dataframe, ScoringError  # noqa: E402

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon="📉", layout="wide")

RISK_COLORS = {"Low": "#2f6b48", "Medium": "#a97527", "High": "#c1621f", "Severe": "#b23b3b"}

st.title("📉 Credit Default Prediction & Portfolio Risk Dashboard")
st.markdown(
    "Turns a table of raw credit-account data into a **portfolio risk view**: a "
    "trained model estimates each account's probability of default (PD), those "
    "estimates roll up into exposure and expected-loss numbers, and the whole "
    "portfolio gets split into risk bands — the same shape of output a risk "
    "committee or lending manager actually reviews. Explore it instantly with "
    "the bundled 30,000-account sample below, or upload your own data in the "
    "sidebar and see it rescored live."
)
st.divider()

st.sidebar.header("Data source")

with st.sidebar.expander("📋 Before you upload — what your CSV needs", expanded=False):
    st.markdown(
        """
One row per credit account, with these columns (original UCI names **or** the
renamed versions below — either works):

| Your column | or this name | What it is |
|---|---|---|
| `LIMIT_BAL` | `credit_limit` | credit line ($) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | `sex`, `education`, `marriage`, `age` | demographics |
| `PAY_0`, `PAY_2`..`PAY_6` | `repay_status_1`..`repay_status_6` | monthly repayment status, most recent first (-1 = paid on time, 1+ = months late) |
| `BILL_AMT1`..`BILL_AMT6` | `bill_amt_1`..`bill_amt_6` | monthly statement balance |
| `PAY_AMT1`..`PAY_AMT6` | `pay_amt_1`..`pay_amt_6` | amount actually paid that month |

**Things to double-check first:**
- It's **raw** account data, not already-scored output — this app does the scoring.
- Every listed column must be present; a missing one stops the upload with a clear error naming what's missing.
- Numeric columns should actually be numbers (no `$`, commas, or blank cells).
- No `default_next_month` / label column needed — this model doesn't need to already know the answer.
- An `ID` (or `account_id`) column is optional; one is generated automatically if missing.

Not sure your file matches? The [UCI 'Default of Credit Card Clients' dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
is exactly this schema and a safe way to test the upload path first.
        """
    )

uploaded = st.sidebar.file_uploader(
    "Upload your own raw credit-account CSV",
    type="csv",
    help="See 'Before you upload' above for the required columns. Leave empty to explore the bundled 30,000-account sample instead.",
)

if uploaded is not None:
    try:
        df = score_raw_dataframe(pd.read_csv(uploaded))
        st.sidebar.success(f"Scored {len(df):,} accounts from your file using the trained model.")
    except ScoringError as e:
        st.sidebar.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Couldn't read that file as CSV: {e}")
        st.stop()
else:
    try:
        df = load_scored_portfolio()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("From the project root, run:\n\n```\npython src/download_data.py\npython src/train_model.py\n```")
        st.stop()

summary = portfolio_summary(df)

st.sidebar.divider()
st.sidebar.download_button(
    "⬇️ Download this scored portfolio as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="scored_portfolio.csv",
    mime="text/csv",
    help="Feed this straight into the GenAI Risk Insights Assistant project to get narrated briefings on this exact data.",
)

# --- KPI row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accounts", f"{summary['n_accounts']:,}")
c2.metric("Total exposure (EAD)", f"${summary['total_exposure']:,.0f}")
c3.metric("Expected loss", f"${summary['expected_loss']:,.0f}")
c4.metric("Expected loss rate", f"{summary['expected_loss_rate']:.2%}")
c5.metric("High/Severe risk share", f"{summary['pct_high_or_severe']:.1%}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Risk band mix")
    band = risk_band_breakdown(df)
    fig = px.bar(
        band, x="risk_band", y="exposure", color="risk_band",
        color_discrete_map=RISK_COLORS, text_auto=".2s",
        labels={"exposure": "Exposure ($)", "risk_band": "Risk band"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(band.style.format({
        "exposure": "${:,.0f}", "expected_loss": "${:,.0f}",
        "avg_pd": "{:.2%}", "pct_of_accounts": "{:.1%}", "pct_of_exposure": "{:.1%}",
    }), use_container_width=True)

with right:
    st.subheader("Expected loss by age segment")
    seg = segment_breakdown(df, "age_band")
    fig2 = px.bar(
        seg, x="age_band", y="expected_loss", color="avg_pd",
        color_continuous_scale="Oranges",
        labels={"expected_loss": "Expected loss ($)", "age_band": "Age band", "avg_pd": "Avg PD"},
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(seg.style.format({
        "exposure": "${:,.0f}", "expected_loss": "${:,.0f}", "avg_pd": "{:.2%}",
    }), use_container_width=True)

st.divider()
st.subheader("Highest-risk accounts (by expected loss)")
n = st.slider("Number of accounts to show", 5, 100, 25)
top = top_risk_accounts(df, n)
st.dataframe(
    top.style.format({
        "credit_limit": "${:,.0f}", "pd_score": "{:.2%}", "expected_loss": "${:,.0f}",
        "utilization": "{:.1%}", "delinquency_trend": "{:+.2f}",
    }).background_gradient(subset=["pd_score"], cmap="Reds"),
    use_container_width=True,
)

with st.expander("How PD / expected loss are calculated"):
    st.markdown(
        """
        - **PD (probability of default)** — model output (XGBoost or logistic regression,
          whichever scored higher on held-out AUC), predicting `default_next_month`.
        - **LGD (loss given default)** — a flat, disclosed assumption of **60%**, not modeled.
          A real risk team would model this separately by product/collateral type.
        - **EAD (exposure at default)** — assumed equal to the account's full credit limit.
        - **Expected loss = PD × LGD × EAD**, summed across the portfolio.

        These are simplifying assumptions, stated openly — the point of this project is to
        demonstrate the *pipeline* (model → PD → portfolio risk rollup → dashboard), not to
        replicate a bank's internal LGD/EAD models.

        **Bring your own data:** upload a CSV in the sidebar and everything on this page —
        KPIs, charts, risk bands, the account table — recomputes live using the model already
        trained on the UCI dataset. Nothing is retrained on your upload; it's scored with the
        existing model, the same way a real risk team would apply a validated model to a new
        batch of accounts.
        """
    )
