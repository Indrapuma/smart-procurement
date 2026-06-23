# CARD-04 · Feature Engineering

**Fase:** 3 — ML Model & XAI  
**Depends on:** CARD-02 (data sudah ada di database)  
**Output:** `ml/feature_matrix.parquet` + `ml/encoders.json` siap untuk training XGBoost

---

## Konteks

XGBoost tidak bisa langsung makan data mentah dari tabel `sales`. Kita perlu:
1. **Join** tabel `sales` + `sales_context` + `products`
2. **Encode** fitur kategorikal (weather, category)
3. **Buat lag features** — XGBoost tidak punya memori waktu seperti LSTM, jadi pola temporal harus kita buat manual sebagai kolom biasa
4. **Simpan encoder mapping** — dibutuhkan lagi saat inference supaya encoding konsisten

---

## Feature Matrix

| Feature | Sumber | Keterangan |
|---|---|---|
| `month` | `sales.date` | 1–12, pola musiman |
| `week_of_year` | `sales.date` | 1–52 |
| `is_holiday_promo` | `sales_context` | 0/1 |
| `weather_encoded` | `sales_context.weather_condition` | label encoded |
| `price` | `sales_context.price_at_sale` | harga saat transaksi |
| `lag_1_week` | `sales.qty_sold` 7 hari lalu | tren jangka pendek |
| `lag_4_week` | `sales.qty_sold` 28 hari lalu | tren jangka menengah |
| `rolling_avg_4w` | moving average 28 hari | sinyal permintaan stabil |
| `rolling_std_4w` | standar deviasi 28 hari | volatilitas permintaan |
| `category_encoded` | `products.category` | label encoded |
| `target` | `sales.qty_sold` | yang diprediksi |

---

## File yang dibuat di card ini

```
ml/
├── features.py
├── feature_matrix.parquet   ← di-generate saat run
└── encoders.json            ← di-generate saat run
```

---

## Script (`ml/features.py`)

```python
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
```

---

## Cara menjalankan

```powershell
# dari root project
uv run ml/features.py
```

Output yang diharapkan:
```
Loading data from database...
Rows   : 58,412
Columns: ['product_id', 'date', 'month', 'week_of_year', ...]
Saved  : ml/feature_matrix.parquet
Saved  : ml/encoders.json
```

> Jumlah baris akan lebih sedikit dari total sales karena baris awal tiap produk di-drop akibat lag features yang NaN.

---

## Kenapa lag features di-shift dulu sebelum rolling?

```python
x.shift(1).rolling(28).mean()
```

`shift(1)` memastikan rolling average tidak mengikutkan hari H sendiri ke dalam perhitungan. Tanpa ini, model akan "melihat masa depan" saat training (*data leakage*).

---

## Troubleshooting

**`relation "sales_context" does not exist`** → Tabel `sales_context` belum dibuat. Jalankan ulang ETL CARD-02 dan pastikan script ETL membuat tabel ini.

**DataFrame kosong setelah `dropna()`** → Produk tidak punya cukup history (minimal 28 hari) untuk lag features. Wajar untuk produk baru.

**`encoders.json` tidak konsisten saat inference** → Jangan re-run `build_feature_matrix()` setelah model di-train. Pakai `encoders.json` yang sudah ada untuk transform data baru.
