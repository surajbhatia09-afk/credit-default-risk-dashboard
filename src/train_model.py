"""
Trains a baseline logistic regression and an XGBoost classifier to predict
default_next_month, compares them, and saves the better one plus a scored
portfolio table for the dashboard.

Usage:
    python src/download_data.py      # once, to get data/raw/credit_clients.csv
    python src/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, add_features

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "credit_clients.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "scored_portfolio.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

LABEL_COL = "default_next_month"


def load_and_prepare() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run `python src/download_data.py` first."
        )
    df = pd.read_csv(RAW_PATH)
    df = add_features(df)
    return df


def train() -> None:
    df = load_and_prepare()
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    # --- Baseline: logistic regression (interpretable, fast, the "explainable" model) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])

    # --- Challenger: XGBoost (usually stronger, still explainable via SHAP) ---
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_ap = average_precision_score(y_test, xgb_proba)

    print(f"Logistic Regression AUC: {logreg_auc:.3f}")
    print(f"XGBoost AUC:             {xgb_auc:.3f}")
    print(f"XGBoost Avg Precision:   {xgb_ap:.3f}")

    winner_name = "xgboost" if xgb_auc >= logreg_auc else "logistic_regression"
    winner = xgb if winner_name == "xgboost" else logreg

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": winner, "scaler": scaler if winner_name == "logistic_regression" else None,
                 "features": FEATURE_COLUMNS, "name": winner_name}, MODEL_PATH)

    metrics = {
        "logistic_regression_auc": round(logreg_auc, 4),
        "xgboost_auc": round(xgb_auc, 4),
        "xgboost_avg_precision": round(xgb_ap, 4),
        "winner": winner_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": round(float(y.mean()), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    # --- Score the FULL portfolio (train+test) for the dashboard ---
    if winner_name == "xgboost":
        df["pd_score"] = winner.predict_proba(X)[:, 1]
    else:
        df["pd_score"] = winner.predict_proba(scaler.transform(X))[:, 1]

    # Simple, disclosed risk assumptions — swap for real LGD/EAD models later
    df["lgd_assumption"] = 0.6          # Loss Given Default: 60% of exposure lost if it defaults
    df["ead"] = df["credit_limit"]       # Exposure at Default: assume full credit line drawn
    df["expected_loss"] = df["pd_score"] * df["lgd_assumption"] * df["ead"]

    df["risk_band"] = pd.cut(
        df["pd_score"],
        bins=[-0.01, 0.05, 0.15, 0.35, 1.01],
        labels=["Low", "Medium", "High", "Severe"],
    )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved scored portfolio ({len(df):,} accounts) -> {PROCESSED_PATH}")


if __name__ == "__main__":
    train()
