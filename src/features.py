"""
Feature engineering for the credit default model.

Turns the raw UCI columns into the kind of derived features an underwriting
or risk-analytics team actually looks at: utilization, payment behaviour
trend, and a delinquency severity score — rather than feeding raw PAY_/
BILL_AMT_ columns straight into the model.
"""

from __future__ import annotations

import pandas as pd

BILL_COLS = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt_{i}" for i in range(1, 7)]
REPAY_COLS = [f"repay_status_{i}" for i in range(1, 7)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Utilization: how much of the credit line is drawn, most recent statement
    df["utilization"] = (df["bill_amt_1"] / df["credit_limit"].replace(0, pd.NA)).clip(0, 5).fillna(0)
    df["avg_utilization_6m"] = (
        df[BILL_COLS].mean(axis=1) / df["credit_limit"].replace(0, pd.NA)
    ).clip(0, 5).fillna(0)

    # Payment ratio: how much of the outstanding bill actually gets paid down
    total_billed = df[BILL_COLS].sum(axis=1).replace(0, pd.NA)
    total_paid = df[PAY_AMT_COLS].sum(axis=1)
    df["payment_ratio_6m"] = (total_paid / total_billed).clip(0, 3).fillna(1.0)

    # Delinquency severity: worst and average repayment-status code over 6 months
    # (UCI encoding: -1 = paid duly, 1 = 1 month late, 2 = 2 months late, ...)
    df["worst_delinquency_6m"] = df[REPAY_COLS].max(axis=1)
    df["avg_delinquency_6m"] = df[REPAY_COLS].mean(axis=1)
    df["months_delinquent_6m"] = (df[REPAY_COLS] > 0).sum(axis=1)

    # Trend: is the customer getting worse or better over the last 3 vs prior 3 months?
    recent = df[["repay_status_1", "repay_status_2", "repay_status_3"]].mean(axis=1)
    prior = df[["repay_status_4", "repay_status_5", "repay_status_6"]].mean(axis=1)
    df["delinquency_trend"] = recent - prior  # positive = getting worse

    df["age_band"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 100], labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )

    return df


FEATURE_COLUMNS = [
    "credit_limit",
    "age",
    "sex",
    "education",
    "marriage",
    "utilization",
    "avg_utilization_6m",
    "payment_ratio_6m",
    "worst_delinquency_6m",
    "avg_delinquency_6m",
    "months_delinquent_6m",
    "delinquency_trend",
]
