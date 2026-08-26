# Credit Default Prediction & Portfolio Risk Dashboard

Predicts the probability of default for individual credit accounts, rolls
those predictions up into portfolio-level risk metrics (exposure, expected
loss, risk-band mix, segment concentration), and surfaces it all in an
interactive Streamlit dashboard.

**Pipeline:** raw data → feature engineering → model training (logistic
regression + XGBoost) → portfolio risk rollup (PD × LGD × EAD) → dashboard.

## Why this project

Built as a portfolio piece demonstrating end-to-end credit-risk analytics:
data engineering, predictive modeling, and translating a model's output into
the kind of exposure/expected-loss view a risk committee actually reads —
the same shape of work behind reporting used in bank/fintech risk teams.

## Quick start

```bash
git clone <your-repo-url>
cd credit-default-risk-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py     # pulls the UCI dataset (see note below)
python src/train_model.py       # trains + scores the portfolio
streamlit run app.py            # opens the dashboard at localhost:8501
```

## Getting the real dataset

`download_data.py` pulls the **UCI "Default of Credit Card Clients"** dataset
(30,000 accounts, 23 features, binary default-next-month label) directly —
no account needed:
<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

If the download fails (network restrictions, UCI temporarily down), the
script automatically falls back to a small synthetic dataset with the same
schema, so the rest of the pipeline still runs — swap in the real file the
moment you're back on an unrestricted connection.

**Want something bigger and messier for a v2?** Two well-known Kaggle
alternatives (need a free Kaggle account + API token — see the step-by-step
guide):
- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — multi-table, ~300k applicants, the "realistic" version
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — single table, ~150k rows, closer in shape to this one

## Bring your own data

The deployed app isn't limited to the bundled sample — open the sidebar and
upload your own raw accounts CSV. It's scored live with the already-trained
model (nothing is retrained on your upload) and every KPI, chart, and table
on the page recomputes from your file.

Your CSV needs these columns (either the original UCI names or the renamed
versions shown here — both work):

| Renamed | Original UCI | Meaning |
|---|---|---|
| `credit_limit` | `LIMIT_BAL` | credit line |
| `sex`, `education`, `marriage`, `age` | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | demographics |
| `repay_status_1..6` | `PAY_0`, `PAY_2..PAY_6` | monthly repayment status, most recent first |
| `bill_amt_1..6` | `BILL_AMT1..6` | monthly statement balance |
| `pay_amt_1..6` | `PAY_AMT1..6` | monthly amount actually paid |

Once scored, use the sidebar's **"Download this scored portfolio as CSV"**
button to feed the result straight into the **GenAI Risk Insights Assistant**
project for narrated briefings on that exact data.

## Project structure

```
credit-default-risk-dashboard/
├── app.py                  # Streamlit dashboard (bundled sample OR your own upload)
├── src/
│   ├── download_data.py    # fetches/generates the raw data
│   ├── features.py         # utilization, payment ratio, delinquency features
│   ├── train_model.py      # trains + compares logistic regression vs XGBoost
│   ├── risk_metrics.py     # portfolio-level rollups (also used by the GenAI project)
│   └── scoring.py          # scores a fresh/uploaded file with the already-trained model
├── notebooks/
│   └── 01_eda.ipynb        # exploratory data analysis
├── data/                   # real UCI dataset + scored output, committed so the deployed app has data to read
├── models/                 # trained model + metrics.json, committed for the same reason
└── requirements.txt
```

## Modeling notes (stated openly, not hidden)

- **LGD (loss given default)** is a flat 60% assumption, not modeled — a real
  risk team would model this separately by product/collateral.
- **EAD (exposure at default)** is assumed to equal the full credit limit.
- The model picks whichever of logistic regression / XGBoost scores higher
  AUC on a held-out test set; both are reported in `models/metrics.json`.

The point of this project is the *pipeline* — model → PD → portfolio
rollup → dashboard — not a from-scratch reproduction of a bank's internal
LGD/EAD models.

## Deploying it live (free)

See the step-by-step guide for pushing this to GitHub and deploying it free
on Streamlit Community Cloud so you have a shareable link, not just a repo.
