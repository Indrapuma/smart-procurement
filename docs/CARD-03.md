# CARD-03 · SQL Tools Implementation

**Fase:** 2 — SQL Tools  
**Depends on:** CARD-02 (data sudah ada di database)  
**Output:** 6 tool functions siap dipanggil oleh LangGraph agent

---

## Konteks

SQL tools adalah jembatan antara agent dan database. Agent **tidak boleh** menulis SQL sendiri — semua query sudah predefined di sini, read-only, dan parameterized. Ini mencegah SQL injection dan query yang tidak terkontrol.

6 tools yang dibutuhkan:

| Tool | Pertanyaan yang dijawab |
|---|---|
| `get_current_stock` | Stok produk X saat ini berapa? |
| `get_sales_history` | Riwayat penjualan produk X N bulan terakhir? |
| `get_supplier_leadtime` | Lead time dan MOQ supplier produk X? |
| `get_demand_trend` | Tren permintaan produk X naik/turun? |
| `get_seasonality_index` | Bulan mana permintaan produk X paling tinggi? |
| `rank_products_by_risk` | Produk mana yang paling berisiko stockout/overstock? |

---

## File yang dibuat di card ini

```
tools/
└── sql_tools.py
```

---

## Script (`tools/sql_tools.py`)

```python
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def _engine():
    return create_engine(DB_URL)


def get_current_stock(product_id: str) -> dict:
    query = text("""
        SELECT p.product_id, p.name, p.category, i.current_stock, i.last_updated
        FROM inventory i
        JOIN products p USING(product_id)
        WHERE i.product_id = :product_id
    """)
    with _engine().connect() as conn:
        row = conn.execute(query, {"product_id": product_id}).fetchone()
    if row is None:
        return {"error": f"Produk {product_id} tidak ditemukan"}
    return {
        "product_id":    row.product_id,
        "name":          row.name,
        "category":      row.category,
        "current_stock": row.current_stock,
        "last_updated":  str(row.last_updated),
    }


def _max_date() -> str:
    with _engine().connect() as conn:
        result = conn.execute(text("SELECT MAX(date) FROM sales"))
        return str(result.scalar())


def get_sales_history(product_id: str, n_months: int = 6) -> dict:
    query = text("""
        SELECT date, qty_sold
        FROM sales
        WHERE product_id = :product_id
          AND date >= (SELECT MAX(date) FROM sales) - (INTERVAL '1 month' * :n_months)
        ORDER BY date
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"product_id": product_id, "n_months": n_months},
                         parse_dates=["date"])
    if df.empty:
        return {"error": f"Tidak ada data penjualan untuk {product_id}"}
    return {
        "product_id": product_id,
        "n_months":   n_months,
        "records":    df.to_dict(orient="records"),
        "total_sold": int(df["qty_sold"].sum()),
        "avg_daily":  round(df["qty_sold"].mean(), 2),
    }


def get_supplier_leadtime(product_id: str) -> dict:
    query = text("""
        SELECT lead_time_days, min_order_qty
        FROM suppliers
        WHERE product_id = :product_id
        LIMIT 1
    """)
    with _engine().connect() as conn:
        row = conn.execute(query, {"product_id": product_id}).fetchone()
    if row is None:
        return {"error": f"Tidak ada data supplier untuk {product_id}"}
    return {
        "product_id":     product_id,
        "lead_time_days": row.lead_time_days,
        "min_order_qty":  row.min_order_qty,
    }


def get_demand_trend(product_id: str, n_months: int = 3) -> dict:
    history = get_sales_history(product_id, n_months=n_months)
    if "error" in history:
        return history

    df = pd.DataFrame(history["records"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Growth rate: bandingkan rata-rata paruh pertama vs paruh kedua periode
    mid = len(df) // 2
    avg_first  = df["qty_sold"].iloc[:mid].mean()
    avg_second = df["qty_sold"].iloc[mid:].mean()
    growth_rate = round((avg_second - avg_first) / avg_first * 100, 2) if avg_first else 0

    # Moving average 7 hari
    df["ma7"] = df["qty_sold"].rolling(7).mean()

    return {
        "product_id":  product_id,
        "n_months":    n_months,
        "growth_rate": growth_rate,          # % positif = naik, negatif = turun
        "avg_first_half":  round(avg_first, 2),
        "avg_second_half": round(avg_second, 2),
        "ma7_latest": round(df["ma7"].dropna().iloc[-1], 2) if not df["ma7"].dropna().empty else None,
    }


def get_seasonality_index(product_id: str) -> dict:
    query = text("""
        SELECT EXTRACT(MONTH FROM date) AS month, AVG(qty_sold) AS avg_qty
        FROM sales
        WHERE product_id = :product_id
        GROUP BY month
        ORDER BY month
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"product_id": product_id})
    if df.empty:
        return {"error": f"Tidak ada data untuk {product_id}"}

    overall_avg = df["avg_qty"].mean()
    df["index"] = (df["avg_qty"] / overall_avg).round(3)

    MONTH_NAMES = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    df["month_name"] = df["month"].astype(int).apply(lambda m: MONTH_NAMES[m - 1])

    peak = df.loc[df["index"].idxmax()]
    return {
        "product_id":   product_id,
        "overall_avg":  round(overall_avg, 2),
        "peak_month":   peak["month_name"],
        "peak_index":   peak["index"],
        "by_month":     df[["month_name", "index"]].to_dict(orient="records"),
    }


def rank_products_by_risk(category: str = None) -> list:
    query = text("""
        SELECT p.product_id, p.name, p.category,
               i.current_stock,
               s.lead_time_days,
               s.min_order_qty
        FROM products p
        JOIN inventory i USING(product_id)
        JOIN suppliers s USING(product_id)
        WHERE (:category IS NULL OR p.category = :category)
        ORDER BY p.product_id
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"category": category})

    # Hitung avg daily sales per produk (seluruh history)
    avg_query = text("""
        SELECT product_id, AVG(qty_sold) AS avg_daily
        FROM sales
        GROUP BY product_id
    """)
    with _engine().connect() as conn:
        avg_df = pd.read_sql(avg_query, conn)

    df = df.merge(avg_df, on="product_id", how="left")
    df["avg_daily"] = df["avg_daily"].fillna(1)

    # Days of stock: berapa hari stok bertahan
    df["days_of_stock"] = (df["current_stock"] / df["avg_daily"]).round(1)

    def classify(row):
        if row["days_of_stock"] < row["lead_time_days"]:
            return "CRITICAL"
        elif row["days_of_stock"] < row["lead_time_days"] * 1.5:
            return "WARNING"
        elif row["current_stock"] > row["avg_daily"] * 90:
            return "OVERSTOCK"
        return "OK"

    RISK_ORDER = {"CRITICAL": 0, "WARNING": 1, "OVERSTOCK": 2, "OK": 3}
    df["risk"] = df.apply(classify, axis=1)
    df = df.sort_values("risk", key=lambda x: x.map(RISK_ORDER))

    return df[["product_id", "name", "category", "current_stock",
               "days_of_stock", "lead_time_days", "risk"]].to_dict(orient="records")
```

---

## Cara test manual

Jalankan dari root project:

```python
# test_tools.py (temporary, hapus setelah selesai)
from tools.sql_tools import (
    get_current_stock,
    get_sales_history,
    get_supplier_leadtime,
    get_demand_trend,
    get_seasonality_index,
    rank_products_by_risk,
)
import json

print(json.dumps(get_current_stock("P0001"), indent=2, default=str))
print(json.dumps(get_supplier_leadtime("P0001"), indent=2))
print(json.dumps(get_demand_trend("P0001", n_months=3), indent=2))
print(json.dumps(get_seasonality_index("P0001"), indent=2))
print(json.dumps(rank_products_by_risk(), indent=2)[:500])  # preview 500 char
```

Output yang diharapkan dari `get_current_stock("P0001")`:
```json
{
  "product_id": "P0001",
  "name": "Groceries Item P0001",
  "category": "Groceries",
  "current_stock": 1050,
  "last_updated": "..."
}
```

Output `rank_products_by_risk()` — produk CRITICAL muncul paling atas:
```json
[
  {"product_id": "P0005", "risk": "CRITICAL", "days_of_stock": 8.2, "lead_time_days": 14, ...},
  {"product_id": "P0012", "risk": "WARNING",  "days_of_stock": 18.5, "lead_time_days": 14, ...},
  ...
]
```

---

## Kenapa tidak pakai LangChain `@tool` decorator sekarang?

Decorator `@tool` akan ditambahkan di CARD-08 (Router Node) ketika tools ini diintegrasikan ke dalam graph. Saat ini fungsi-fungsi ini berdiri sendiri agar mudah di-test secara independen.

---

## Troubleshooting

**`KeyError` atau `None` dari query** → Cek apakah ETL (CARD-02) sudah dijalankan dan data ada di DB.

**`CURRENT_DATE` tidak cocok dengan data** → Dataset berakhir di 2024-01-01, sedangkan `CURRENT_DATE` adalah hari ini (2026). Akibatnya `get_sales_history` dengan filter `n_months` dari hari ini akan return kosong. Fix: ganti `CURRENT_DATE` dengan tanggal max di data.
