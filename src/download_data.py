"""
Downloads the UCI 'Default of Credit Card Clients' dataset (30,000 accounts,
23 features, binary default-next-month label) — no login required.

Source: UCI Machine Learning Repository, dataset #350
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

If the download fails (e.g. you're behind a restrictive network, or UCI is
temporarily down), this falls back to generating a small synthetic dataset
with the same schema so the rest of the pipeline still runs. Swap in the
real file — or a Kaggle dataset — the moment you can; see README.md for the
Kaggle "Home Credit Default Risk" / "Give Me Some Credit" alternatives if
you want a bigger, messier, more realistic dataset for a v2.

Usage:
    python src/download_data.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = RAW_DIR / "credit_clients.csv"

UCI_XLS_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

COLUMN_RENAME = {
    "ID": "account_id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "repay_status_1",
    "PAY_2": "repay_status_2",
    "PAY_3": "repay_status_3",
    "PAY_4": "repay_status_4",
    "PAY_5": "repay_status_5",
    "PAY_6": "repay_status_6",
    "BILL_AMT1": "bill_amt_1",
    "BILL_AMT2": "bill_amt_2",
    "BILL_AMT3": "bill_amt_3",
    "BILL_AMT4": "bill_amt_4",
    "BILL_AMT5": "bill_amt_5",
    "BILL_AMT6": "bill_amt_6",
    "PAY_AMT1": "pay_amt_1",
    "PAY_AMT2": "pay_amt_2",
    "PAY_AMT3": "pay_amt_3",
    "PAY_AMT4": "pay_amt_4",
    "PAY_AMT5": "pay_amt_5",
    "PAY_AMT6": "pay_amt_6",
    "default payment next month": "default_next_month",
    "default.payment.next.month": "default_next_month",
}


def try_download() -> pd.DataFrame | None:
    try:
        print(f"Downloading from {UCI_XLS_URL} ...")
        resp = requests.get(UCI_XLS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import zipfile

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xls_name = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
            with zf.open(xls_name) as f:
                df = pd.read_excel(f, header=1)
        print(f"Downloaded {len(df):,} rows.")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed ({exc}). Falling back to a synthetic sample dataset.")
        return None


def make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Small synthetic stand-in with the same schema, so the pipeline still runs."""
    rng = np.random.default_rng(seed)
    limit_bal = rng.choice([20000, 50000, 90000, 150000, 200000, 360000], size=n)
    age = rng.integers(21, 65, size=n)
    sex = rng.integers(1, 3, size=n)
    education = rng.integers(1, 5, size=n)
    marriage = rng.integers(1, 4, size=n)

    repay_cols = {f"PAY_{i}" if i > 0 else "PAY_0": rng.integers(-1, 5, size=n) for i in [0, 2, 3, 4, 5, 6]}
    bill_cols = {f"BILL_AMT{i}": (rng.random(n) * limit_bal * 0.6).round(2) for i in range(1, 7)}
    pay_cols = {f"PAY_AMT{i}": (rng.random(n) * limit_bal * 0.15).round(2) for i in range(1, 7)}

    # Rough, illustrative default probability driven by repayment status + utilization
    util = bill_cols["BILL_AMT1"] / np.maximum(limit_bal, 1)
    worst_repay = np.maximum.reduce([v for k, v in repay_cols.items()])
    logit = -2.2 + 0.55 * worst_repay + 1.8 * util
    prob = 1 / (1 + np.exp(-logit))
    default_next_month = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **repay_cols,
            **bill_cols,
            **pay_cols,
            "default payment next month": default_next_month,
        }
    )
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = try_download()
    synthetic = df is None
    if df is None:
        df = make_synthetic()

    df = df.rename(columns=COLUMN_RENAME)
    df.to_csv(OUT_PATH, index=False)

    tag = "SYNTHETIC sample" if synthetic else "real UCI"
    print(f"Saved {len(df):,} rows ({tag} data) to {OUT_PATH}")
    if synthetic:
        print(
            "\nNOTE: this is a synthetic placeholder, not the real dataset.\n"
            "Download it yourself from the UCI page or Kaggle mirror and drop it at\n"
            f"  {OUT_PATH}\n"
            "See README.md > 'Getting the real dataset' for both links.\n"
        )


if __name__ == "__main__":
    sys.exit(main())
