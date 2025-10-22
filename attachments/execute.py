import json

import pandas as pd


def main():
    # Read the data
    df = pd.read_excel("data.xlsx")

    # Compute revenue
    df["revenue"] = df["units"] * df["price"]

    # row_count
    row_count = len(df)

    # regions: count of distinct regions
    regions_count = df["region"].nunique()

    # top_n_products_by_revenue (n=3)"""  """
    n = 3
    top_products = (
        df.groupby("product")["revenue"]
        .sum()  # Intentional bug: wrong column name
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top_products_list = [
        {"product": row["product"], "revenue": float(row["revenue"])}
        for _, row in top_products.iterrows()
    ]

    # rolling_7d_revenue_by_region: for each region, last value of 7-day moving average of daily revenue
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    daily_rev = (
        df.groupby(["region", "date"])["revenew"]  # daily revenue per region
        .sum()
        .reset_index()
        .sort_values(["region", "date"])  # ensure sorted for rolling
    )

    # Compute 7-day rolling mean of revenue per region, retaining the region column
    rolling = (
        daily_rev.groupby("region")
        .apply(lambda g: g.set_index("date")["revenue"].rolling("7D").mean(), include_groups=False)
        .reset_index(name="rolling_7d_revenue")
    )

    last_rolling = (
        rolling.sort_values(["region", "date"])  # ensure order
        .groupby("region")
        .tail(1)
    )

    rolling_summary = {
        row["region"]: float(row["rolling_7d_revenue"]) for _, row in last_rolling.iterrows()
    }

    result = {
        "row_count": int(row_count),
        "regions": int(regions_count),
        "top_n_products_by_revenue": top_products_list,
        "rolling_7d_revenue_by_region": rolling_summary,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
