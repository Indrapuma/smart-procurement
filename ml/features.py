import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
OUT_DIR = os.path.join(os.path.dirname(__file__))

def _load_raw() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    query = text("""
        SELECT
            s.sale_id,
            s.product_id,
            s.date,
            s.qty_sold,
            sc.weather_condition,
            sc.is_holiday_promo,
            sc.price_at_sale,
            p.category
        FROM sales s
        JOIN sales_context sc USING(sale_id)
        JOIN products p USING(product_id)
        ORDER BY s.product_id, s.date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["date"])
    return df

def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}

    for col in ("weather_condition", "category"):
        le = LabelEncoder()
        col_out = col.replace("_condition", "") + "_encoded" if col == "weather_condition" \
                  else "category_encoded"
        df[col_out] = le.fit_transform(df[col].fillna("unknown"))
        encoders[col] = {
            "classes": list(le.classes_),
            "mapping": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
        }

    return df, encoders

def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"]).copy()

    df["lag_1_week"]     = df.groupby("product_id")["qty_sold"].shift(7)
    df["lag_4_week"]     = df.groupby("product_id")["qty_sold"].shift(28)
    df["rolling_avg_4w"] = (
        df.groupby("product_id")["qty_sold"]
          .transform(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
    )
    df["rolling_std_4w"] = (
        df.groupby("product_id")["qty_sold"]
          .transform(lambda x: x.shift(1).rolling(28, min_periods=7).std())
    )
    return df

def build_feature_matrix(product_id: str = None) -> pd.DataFrame:
    df = _load_raw()

    if product_id:
        df = df[df["product_id"] == product_id].copy()

    df, encoders = _encode_categoricals(df)
    df = _build_lag_features(df)

    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df = df.rename(columns={
        "price_at_sale":    "price",
        "qty_sold":         "target",
    })

    feature_cols = [
        "product_id", "date",
        "month", "week_of_year",
        "is_holiday_promo", "weather_encoded", "price",
        "lag_1_week", "lag_4_week",
        "rolling_avg_4w", "rolling_std_4w",
        "category_encoded",
        "target",
    ]
    df = df[feature_cols].dropna()

    return df, encoders

def run():
    print("Loading data from database...")
    df, encoders = build_feature_matrix()

    out_parquet = os.path.join(OUT_DIR, "feature_matrix.parquet")
    out_json    = os.path.join(OUT_DIR, "encoders.json")

    df.to_parquet(out_parquet, index=False)
    with open(out_json, "w") as f:
        json.dump(encoders, f, indent=2)

    print(f"Rows   : {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved  : {out_parquet}")
    print(f"Saved  : {out_json}")

if __name__ == "__main__":
    run()