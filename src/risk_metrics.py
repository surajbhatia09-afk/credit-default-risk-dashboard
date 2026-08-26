"""
Rolls up the account-level scored portfolio (from train_model.py) into
portfolio-level risk metrics — the numbers a risk committee / manager
actually wants: total exposure, expected loss, risk-band mix, concentration
by segment, and month-over-month style deltas (simulated by re-scoring
periodically once you're comfortable extending this).

Used by app.py (the dashboard) AND by the GenAI Risk Insights Assistant
project, which reads scored_portfolio.csv as its data source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCORED_PATH = ROOT / "data" / "processed" / "scored_portfolio.csv"


def load_scored_portfolio() -> pd.DataFrame:
    if not SCORED_PATH.exists():
        raise FileNotFoundError(
            f"{SCORED_PATH} not found. Run `python src/train_model.py` first."
        )
    df = pd.read_csv(SCORED_PATH)
    return df


def portfolio_summary(df: pd.DataFrame) -> dict:
    return {
        "n_accounts": int(len(df)),
        "total_exposure": float(df["ead"].sum()),
        "expected_loss": float(df["expected_loss"].sum()),
        "expected_loss_rate": float(df["expected_loss"].sum() / max(df["ead"].sum(), 1)),
        "avg_pd": float(df["pd_score"].mean()),
        "pct_high_or_severe": float(df["risk_band"].isin(["High", "Severe"]).mean()),
    }


def risk_band_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("risk_band", observed=True)
        .agg(
            accounts=("account_id", "count"),
            exposure=("ead", "sum"),
            expected_loss=("expected_loss", "sum"),
            avg_pd=("pd_score", "mean"),
        )
        .reset_index()
    )
    out["pct_of_accounts"] = out["accounts"] / out["accounts"].sum()
    out["pct_of_exposure"] = out["exposure"] / out["exposure"].sum()
    return out


def segment_breakdown(df: pd.DataFrame, segment_col: str = "age_band") -> pd.DataFrame:
    out = (
        df.groupby(segment_col, observed=True)
        .agg(
            accounts=("account_id", "count"),
            exposure=("ead", "sum"),
            expected_loss=("expected_loss", "sum"),
            avg_pd=("pd_score", "mean"),
        )
        .reset_index()
        .sort_values("expected_loss", ascending=False)
    )
    return out


def top_risk_accounts(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    cols = [
        "account_id", "credit_limit", "pd_score", "risk_band",
        "expected_loss", "worst_delinquency_6m", "utilization", "delinquency_trend",
    ]
    return df.sort_values("expected_loss", ascending=False)[cols].head(n)
