python
#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _read_data() -> tuple[pd.DataFrame, str]:
    """
    Read input data from data.csv if present; otherwise fall back to data.xlsx.
    If reading from Excel, also export a normalized CSV for reproducibility.
    Returns:
        (df, source_file)
    """
    csv_path = Path("data.csv")
    xlsx_path = Path("data.xlsx")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        src = "data.csv"
    elif xlsx_path.exists():
        # openpyxl is required only when reading Excel; CI installs it.
        df = pd.read_excel(xlsx_path)
        # Normalize to CSV for future runs (not committed automatically).
        # This helps local runs remain engine-free.
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            pass
        src = "data.xlsx"
    else:
        raise FileNotFoundError(
            "No input found. Provide data.csv (preferred) or data.xlsx."
        )

    return df, src


def _compute_top_products(df: pd.DataFrame, n: int = 3) -> List[Dict[str, float]]:
    top_products = (
        df.groupby("product", dropna=False)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    return [
        {"product": str(row["product"]), "revenue": float(row["revenue"])}
        for _, row in top_products.iterrows()
    ]


def _rolling_7d_ma_by_region(df: pd.DataFrame) -> Dict[str, float]:
    """
    For each region, compute the last value of the 7-day moving average of daily revenue.
    Days with no sales are treated as revenue=0 within the window.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        # Drop rows without a valid date; they cannot contribute to daily aggregates
        df = df.dropna(subset=["date"])

    # Aggregate revenue per region per calendar day
    daily = (
        df.groupby(["region", pd.Grouper(key="date", freq="D")], dropna=False)["revenue"]
        .sum()
        .reset_index()
    )

    results: Dict[str, float] = {}
    for region, grp in daily.groupby("region", dropna=False):
        # Build a continuous date index from min to max; fill missing with 0
        if grp.empty:
            results[str(region)] = 0.0
            continue
        grp = grp.set_index("date").sort_index()
        idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        s = grp["revenue"].reindex(idx, fill_value=0.0)
        ma7 = s.rolling(window=7, min_periods=1).mean()
        results[str(region)] = float(ma7.iloc[-1])

    return results


def main() -> None:
    df, source_file = _read_data()

    # Ensure numeric columns
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Compute revenue
    df["revenue"] = df["units"] * df["price"]

    # Basic stats
    row_count = int(len(df))
    regions_count = int(df["region"].nunique(dropna=True))

    # Top N products by revenue
    top_products_list = _compute_top_products(df, n=3)

    # Rolling 7-day MA of daily revenue per region (last value)
    rolling_ma = _rolling_7d_ma_by_region(df)

    result = {
        "row_count": row_count,
        "regions_count": regions_count,
        "top_n_products_by_revenue": top_products_list,
        "rolling_7d_revenue_by_region": rolling_ma,
        "source_file": source_file,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    json.dump(result, fp=None)  # type: ignore[call-arg]


if __name__ == "__main__":
    # json.dump requires a file-like if fp is provided; we want stdout.
    # So we use print(json.dumps(...)) to ensure stdout output.
    df, src = _read_data()
    # Move the core logic here to avoid duplicating work in main()
    # Recompute with the same logic as in main(), then print to stdout.
    # This preserves the requested `python execute.py > result.json` behavior.

    # Ensure numeric columns
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Compute revenue
    df["revenue"] = df["units"] * df["price"]

    # Basic stats
    row_count = int(len(df))
    regions_count = int(df["region"].nunique(dropna=True))

    # Top N products by revenue
    top_products_list = _compute_top_products(df, n=3)

    # Rolling 7-day MA of daily revenue per region (last value)
    rolling_ma = _rolling_7d_ma_by_region(df)

    payload = {
        "row_count": row_count,
        "regions_count": regions_count,
        "top_n_products_by_revenue": top_products_list,
        "rolling_7d_revenue_by_region": rolling_ma,
        "source_file": src,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    print(json.dumps(payload, indent=2))