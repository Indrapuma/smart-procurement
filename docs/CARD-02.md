# CARD-02 · ETL Pipeline (Dataset → PostgreSQL)

**Fase:** 1 — Database & Data Pipeline  
**Depends on:** CARD-01 (schema sudah dibuat)  
**Output:** ~73K baris data masuk ke 4 tabel PostgreSQL + tabel `sales_context`

---

## Konteks

Dataset Kaggle yang kita punya (`retail-store-inventory-forecasting-dataset.zip`) punya struktur yang **berbeda** dari schema internal kita. Tugas ETL adalah menjembatani keduanya.

**Struktur dataset asli (1 baris = 1 produk di 1 toko di 1 hari):**
```
Date | Store ID | Product ID | Category | Inventory Level | Units Sold | Price | Weather Condition | Holiday/Promotion | ...
2022-01-01 | S001 | P0001 | Groceries | 231 | 127 | 33.50 | Rainy | 0 | ...
2022-01-01 | S002 | P0001 | Groceries | 189 | 98  | 33.50 | Sunny | 0 | ...
```

**Yang kita butuhkan (agregat per produk per hari):**
```
sales: product_id=P0001, date=2022-01-01, qty_sold=225  ← sum dari semua toko
inventory: product_id=P0001, current_stock=2100         ← stok terbaru, total semua toko
```

**Stats dataset:**
- 73.100 baris total
- 20 produk unik (P0001–P0020)
- 5 toko (S001–S005)
- 5 kategori: Groceries, Toys, Electronics, Furniture, Clothing
- Rentang tanggal: 2022-01-01 s/d 2024-01-01

---

## Prerequisites

1. PostgreSQL sudah berjalan lokal
2. Database `procurement` sudah dibuat
3. Migration CARD-01 sudah dijalankan (`001_create_tables.sql`)
4. File `retail-store-inventory-forecasting-dataset.zip` ada di root project
5. Dependencies terinstall: `pip install sqlalchemy psycopg2-binary pandas numpy python-dotenv`

Cek koneksi database sebelum mulai:
```bash
psql -U postgres -d procurement -c "\dt"
# Harus tampil 4 tabel: products, sales, inventory, suppliers
```

---

## File yang dibuat di card ini

```
data/
└── etl.py          ← script utama ETL
db/
└── 002_add_sales_context.sql   ← migration tabel tambahan
```

---

## Step 1 — Tambah Tabel `sales_context`

Tabel ini menyimpan konteks per transaksi (cuaca, promo, harga) yang dibutuhkan SHAP nanti.  
Buat file `db/002_add_sales_context.sql`:

```sql
CREATE TABLE IF NOT EXISTS sales_context (
    sale_id             INTEGER PRIMARY KEY REFERENCES sales(sale_id),
    weather_condition   VARCHAR(50),
    is_holiday_promo    BOOLEAN NOT NULL DEFAULT FALSE,
    price_at_sale       NUMERIC(10,2),
    discount_pct        NUMERIC(5,2),
    competitor_pricing  NUMERIC(10,2),
    seasonality         VARCHAR(20)
);
```

Jalankan:
```bash
psql -U postgres -d procurement -f db/002_add_sales_context.sql
```

---

## Step 2 — Script ETL (`data/etl.py`)

Simpan script berikut sebagai `data/etl.py`:

```python
"""
ETL Pipeline: retail_store_inventory.csv → PostgreSQL
Mengubah data flat (per toko per hari) menjadi schema internal procurement agent.
"""

import os
import zipfile
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

ZIP_PATH = "retail-store-inventory-forecasting-dataset.zip"
CSV_NAME = "retail_store_inventory.csv"
DB_URL   = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/procurement")

# Lead time sintetis per kategori (hari): (min, max)
LEAD_TIME_RANGE = {
    "Groceries":   (3,  7),
    "Toys":        (7,  14),
    "Electronics": (14, 21),
    "Furniture":   (14, 21),
    "Clothing":    (7,  14),
}

# Min order qty sintetis (pilihan realistis per kategori)
MIN_ORDER_OPTIONS = {
    "Groceries":   [10, 20, 50],
    "Toys":        [5,  10, 20],
    "Electronics": [1,  5,  10],
    "Furniture":   [1,  2,  5],
    "Clothing":    [10, 20, 50],
}

np.random.seed(42)  # reproducible

# ─── Extract ──────────────────────────────────────────────────────────────────

def extract() -> pd.DataFrame:
    """Load CSV dari zip file tanpa perlu extract manual."""
    print("▶ Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(CSV_NAME) as f:
            df = pd.read_csv(f, parse_dates=["Date"])
    print(f"  ✓ {len(df):,} baris, {df['Product ID'].nunique()} produk, {df['Store ID'].nunique()} toko")
    return df


# ─── Transform ────────────────────────────────────────────────────────────────

def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    20 produk unik. Dataset tidak punya kolom 'name' eksplisit,
    jadi kita generate dari category + product_id.
    Harga diambil dari rata-rata price di seluruh data per produk.
    """
    print("▶ Transforming products...")
    products = (
        df.groupby("Product ID")
        .agg(category=("Category", "first"), unit_price=("Price", "mean"))
        .reset_index()
        .rename(columns={"Product ID": "product_id"})
    )
    # Generate nama yang lebih human-readable
    products["name"] = products["category"] + " Item " + products["product_id"]
    products["unit_price"] = products["unit_price"].round(2)
    print(f"  ✓ {len(products)} produk")
    return products[["product_id", "name", "category", "unit_price"]]


def transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregat Units Sold per Product ID + Date (sum dari semua toko).
    1 baris per produk per hari — ini yang akan dipakai model ML.
    """
    print("▶ Transforming sales...")
    sales = (
        df.groupby(["Product ID", "Date"])["Units Sold"]
        .sum()
        .reset_index()
        .rename(columns={"Product ID": "product_id", "Date": "date", "Units Sold": "qty_sold"})
    )
    print(f"  ✓ {len(sales):,} baris sales (produk × hari)")
    return sales


def transform_sales_context(df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Konteks per hari per produk (agregat dari semua toko):
    - weather: modus (cuaca yang paling banyak terjadi hari itu)
    - is_holiday_promo: 1 jika ada toko yang promosi hari itu
    - price: rata-rata harga jual
    - discount: rata-rata diskon
    - competitor_pricing: rata-rata harga kompetitor
    - seasonality: sama untuk semua toko (ambil first)
    """
    print("▶ Transforming sales_context...")
    context = (
        df.groupby(["Product ID", "Date"])
        .agg(
            weather_condition=("Weather Condition", lambda x: x.mode()[0]),
            is_holiday_promo=("Holiday/Promotion", "max"),
            price_at_sale=("Price", "mean"),
            discount_pct=("Discount", "mean"),
            competitor_pricing=("Competitor Pricing", "mean"),
            seasonality=("Seasonality", "first"),
        )
        .reset_index()
        .rename(columns={"Product ID": "product_id", "Date": "date"})
    )

    # Join dengan sales_df untuk dapat sale_id
    merged = sales_df.merge(context, on=["product_id", "date"], how="left")
    merged["is_holiday_promo"] = merged["is_holiday_promo"].astype(bool)
    merged["price_at_sale"] = merged["price_at_sale"].round(2)
    merged["discount_pct"] = merged["discount_pct"].round(2)
    merged["competitor_pricing"] = merged["competitor_pricing"].round(2)
    print(f"  ✓ {len(merged):,} baris sales_context")
    return merged[["sale_id", "weather_condition", "is_holiday_promo",
                   "price_at_sale", "discount_pct", "competitor_pricing", "seasonality"]]


def transform_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stok 'saat ini' = Inventory Level di tanggal paling akhir, dijumlah dari semua toko.
    Ini simulasi snapshot stok terkini.
    """
    print("▶ Transforming inventory...")
    latest_date = df["Date"].max()
    inventory = (
        df[df["Date"] == latest_date]
        .groupby("Product ID")["Inventory Level"]
        .sum()
        .reset_index()
        .rename(columns={"Product ID": "product_id", "Inventory Level": "current_stock"})
    )
    print(f"  ✓ {len(inventory)} produk, tanggal snapshot: {latest_date.date()}")
    return inventory


def transform_suppliers(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lead time dan MOQ di-generate sintetis per kategori.
    Realitanya data ini harus dari sistem ERP/supplier — ini hanya untuk demo.
    """
    print("▶ Transforming suppliers (synthetic)...")
    records = []
    for _, row in products_df.iterrows():
        lo, hi = LEAD_TIME_RANGE.get(row["category"], (7, 14))
        lead_time = int(np.random.randint(lo, hi + 1))
        min_order = int(np.random.choice(MIN_ORDER_OPTIONS.get(row["category"], [10])))
        records.append({
            "product_id":     row["product_id"],
            "lead_time_days": lead_time,
            "min_order_qty":  min_order,
        })
    suppliers = pd.DataFrame(records)
    print(f"  ✓ {len(suppliers)} supplier records")
    return suppliers


# ─── Load ─────────────────────────────────────────────────────────────────────

def load(products, sales, inventory, suppliers) -> pd.DataFrame:
    """
    Load ke PostgreSQL menggunakan SQLAlchemy.
    Urutan insert penting karena ada foreign key:
    products → sales → inventory, suppliers

    Return: sales_df dengan kolom sale_id (generated oleh DB).
    """
    print("▶ Loading ke database...")
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        # Truncate dulu supaya bisa re-run tanpa duplikat
        print("  Truncating existing data...")
        conn.execute(text("TRUNCATE TABLE sales_context, sales, inventory, suppliers, products RESTART IDENTITY CASCADE"))

        # Load products
        products.to_sql("products", conn, if_exists="append", index=False)
        print(f"  ✓ {len(products)} products loaded")

        # Load sales — to_sql tidak return IDs, jadi kita fetch setelah insert
        sales.to_sql("sales", conn, if_exists="append", index=False)
        print(f"  ✓ {len(sales):,} sales loaded")

        # Load inventory
        inventory.to_sql("inventory", conn, if_exists="append", index=False)
        print(f"  ✓ {len(inventory)} inventory records loaded")

        # Load suppliers
        suppliers.to_sql("suppliers", conn, if_exists="append", index=False)
        print(f"  ✓ {len(suppliers)} supplier records loaded")

    # Fetch sales dengan sale_id untuk context join
    with engine.connect() as conn:
        sales_with_id = pd.read_sql(
            "SELECT sale_id, product_id, date FROM sales ORDER BY sale_id",
            conn,
            parse_dates=["date"]
        )

    print(f"  ✓ Fetched {len(sales_with_id):,} sale_ids")
    return sales_with_id


def load_context(sales_context_df: pd.DataFrame) -> None:
    """Load sales_context setelah sales sudah punya sale_id."""
    print("▶ Loading sales_context...")
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        sales_context_df.to_sql("sales_context", conn, if_exists="append", index=False)
    print(f"  ✓ {len(sales_context_df):,} context records loaded")


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify() -> None:
    """Quick sanity check: tampilkan row count tiap tabel."""
    print("\n▶ Verifying...")
    engine = create_engine(DB_URL)
    tables = ["products", "sales", "inventory", "suppliers", "sales_context"]
    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table:<20} {count:>8,} rows")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 50)
    print("ETL Pipeline — Procurement Agent")
    print("=" * 50)

    # Extract
    raw_df = extract()

    # Transform
    products_df  = transform_products(raw_df)
    sales_df     = transform_sales(raw_df)
    inventory_df = transform_inventory(raw_df)
    suppliers_df = transform_suppliers(products_df)

    # Load (urutan penting: products dulu)
    sales_with_id = load(products_df, sales_df, inventory_df, suppliers_df)

    # Transform & load context (butuh sale_id dari DB)
    context_df = transform_sales_context(raw_df, sales_with_id)
    load_context(context_df)

    # Verify
    verify()

    print("\n✓ ETL selesai.")


if __name__ == "__main__":
    run()
```

---

## Step 3 — Jalankan ETL

Pastikan `.env` sudah ada dengan `DATABASE_URL`:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/procurement
```

Jalankan dari root project:
```bash
python data/etl.py
```

Output yang diharapkan:
```
==================================================
ETL Pipeline — Procurement Agent
==================================================
▶ Extracting dataset...
  ✓ 73,100 baris, 20 produk, 5 toko
▶ Transforming products...
  ✓ 20 produk
▶ Transforming sales...
  ✓ 14,620 baris sales (produk × hari)
▶ Transforming inventory...
  ✓ 20 produk, tanggal snapshot: 2024-01-01
▶ Transforming suppliers (synthetic)...
  ✓ 20 supplier records
▶ Loading ke database...
  Truncating existing data...
  ✓ 20 products loaded
  ✓ 14,620 sales loaded
  ✓ 20 inventory records loaded
  ✓ 20 supplier records loaded
  ✓ Fetched 14,620 sale_ids
▶ Loading sales_context...
  ✓ 14,620 context records loaded

▶ Verifying...
  products              20 rows
  sales             14,620 rows
  inventory             20 rows
  suppliers             20 rows
  sales_context     14,620 rows

✓ ETL selesai.
```

---

## Step 4 — Verifikasi Manual di PostgreSQL

Cek beberapa query untuk pastikan data masuk dengan benar:

```sql
-- 1. Lihat sample produk
SELECT * FROM products LIMIT 5;

-- 2. Total penjualan per produk (harus 731 hari × 20 produk = 14.620 baris)
SELECT product_id, COUNT(*) as days, SUM(qty_sold) as total_sold
FROM sales
GROUP BY product_id
ORDER BY total_sold DESC;

-- 3. Stok terkini
SELECT p.name, i.current_stock, s.lead_time_days
FROM inventory i
JOIN products p USING(product_id)
JOIN suppliers s USING(product_id)
ORDER BY i.current_stock;

-- 4. Cek context tersambung dengan benar
SELECT s.product_id, s.date, s.qty_sold, sc.weather_condition, sc.is_holiday_promo
FROM sales s
JOIN sales_context sc USING(sale_id)
WHERE s.product_id = 'P0001'
LIMIT 5;
```

---

## Kenapa kita agregat per produk, bukan per toko?

Dataset punya 5 toko, tapi agent kita query per `product_id` tanpa membedakan toko.  
Ada dua pendekatan:

| Pendekatan | Pro | Cons |
|---|---|---|
| **Agregat per produk+hari** ← yang kita pakai | Schema lebih simpel, model belajar demand total | Kehilangan variasi antar toko |
| Per toko+produk+hari | Data lebih granular | Schema jadi lebih kompleks, agent harus tahu store_id |

Untuk demo portfolio, agregat sudah cukup dan lebih mudah dijelaskan.

---

## Troubleshooting

**Error: `TRUNCATE ... RESTART IDENTITY CASCADE` gagal**  
→ Pastikan urutan tabel di TRUNCATE: `sales_context` dulu baru `sales`, karena ada foreign key.

**Error: `psycopg2.OperationalError: could not connect to server`**  
→ Pastikan PostgreSQL berjalan: `pg_ctl status` atau cek di Services (Windows).

**Error: `UndefinedTable: relation "sales_context" does not exist`**  
→ Jalankan dulu `db/002_add_sales_context.sql`.

**Script berjalan tapi data 0 rows**  
→ Cek apakah `ZIP_PATH` benar — script harus dijalankan dari root project, bukan dari folder `data/`.
