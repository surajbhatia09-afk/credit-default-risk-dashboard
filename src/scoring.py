"""
Shared scoring logic: turns a raw (unscored) credit-accounts table into the
same scored-portfolio schema app.py and the GenAI Risk Insights Assistant
project both expect (pd_score, risk_band, ead, expected_loss, ...).

Used by app.py so a visitor can upload their OWN raw data and get a real,
freshly-scored dashboard using the already-trained model — not just the
bundled UCI sample.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, add_features
from download_data import COLUMN_RENAME

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"

# The minimum raw fields needed to compute every engineered feature. Accepts
# either the original UCI column names (LIMIT_BAL, PAY_0, BILL_AMT1, ...) or
# this project's renamed versions (credit_limit, repay_status_1, ...) —
# whichever your file already uses.
REQUIRED_RAW_FIELDS_RENAMED = [
    "credit_limit", "age", "sex", "education", "marriage",
    *[f"repay_status_{i}" for i in range(1, 7)],
    *[f"bill_amt_{i}" for i in range(1, 7)],
    *[f"pay_amt_{i}" for i in range(1, 7)],
]


class ScoringError(ValueError):
    """Raised when an uploaded file can't be scored — message is shown to the user as-is."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames original UCI columns to this project's names if present; leaves
    already-renamed columns alone."""
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_model():
    if not MODEL_PATH.exists():
        raise ScoringError(
            f"No trained model found at {MODEL_PATH}. Run `python src/train_model.py` once "
            "(or use the bundled sample data) before scoring your own file."
        )
    return joblib.load(MODEL_PATH)


def score_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw uploaded accounts -> fully scored portfolio, using the model already
    trained and shipped in models/model.pkl. Raises ScoringError with a
    human-readable message on anything an end user needs to fix."""
    df = _normalize_columns(raw_df.copy())

    missing = [c for c in REQUIRED_RAW_FIELDS_RENAMED if c not in df.columns]
    if missing:
        raise ScoringError(
            "Your file is missing columns this model needs: " + ", ".join(missing) + ". "
            "Expected either the original UCI 'Default of Credit Card Clients' columns "
            "(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0..PAY_6, BILL_AMT1..6, PAY_AMT1..6) "
            "or this project's renamed versions — see the README for the full column list."
        )

    if "account_id" not in df.columns:
        id_col = "ID" if "ID" in df.columns else None
        df["account_id"] = df[id_col] if id_col else range(1, len(df) + 1)

    df = add_features(df)

    bundle = load_model()
    model, scaler, features, name = bundle["model"], bundle["scaler"], bundle["features"], bundle["name"]
    X = df[features]
    if name == "logistic_regression":
        df["pd_score"] = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        df["pd_score"] = model.predict_proba(X)[:, 1]

    df["lgd_assumption"] = 0.6
    df["ead"] = df["credit_limit"]
    df["expected_loss"] = df["pd_score"] * df["lgd_assumption"] * df["ead"]
    df["risk_band"] = pd.cut(
        df["pd_score"], bins=[-0.01, 0.05, 0.15, 0.35, 1.01],
        labels=["Low", "Medium", "High", "Severe"],
    )
    return df
