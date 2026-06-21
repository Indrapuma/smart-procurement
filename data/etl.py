import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "dataset/retail_store_inventory.csv"
DB_URL = os.getenv("DATABASE_URL")

LEAD_TIME_RANGE = {
    "Groceries":   (3,  7),
    "Toys":        (7,  14),
    "Electronics": (14, 21),
    "Furniture":   (14, 21),
    "Clothing":    (7,  14),
}

MIN_ORDER_OPTIONS = {
    "Groceries":   [10, 20, 50],
    "Toys":        [5,  10, 20],
    "Electronics": [1,  5,  10],
    "Furniture":   [1,  2,  5],
    "Clothing":    [10, 20, 50],
}

np.random.seed(42)


def extract() -> pd.DataFrame:
    print("▶ Extracting dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    print(f"  ✓ {len(df):,} baris, {df['Product ID'].nunique()} produk, {df['Store ID'].nunique()} toko")
    return df


def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    print("▶ Transforming products...")
    products = (
        df.groupby("Product ID")
        .agg(category=("Category", "first"), unit_price=("Price", "mean"))
        .reset_index()
        .rename(columns={"Product ID": "product_id"})
    )
    products["name"] = products["category"] + " Item " + products["product_id"]
    products["unit_price"] = products["unit_price"].round(2)
    print(f"  ✓ {len(products)} produk")
    return products[["product_id", "name", "category", "unit_price"]]


def transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    print("▶ Transforming sales...")
    sales = (
        df.groupby(["Product ID", "Date"])["Units Sold"]
        .sum()
        .reset_index()
        .rename(columns={"Product ID": "product_id", "Date": "date", "Units Sold": "qty_sold"})
    )
    print(f"  ✓ {len(sales):,} baris sales (produk × hari)")
    return sales


def transform_inventory(df: pd.DataFrame) -> pd.DataFrame:
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
    print("▶ Transforming suppliers (synthetic)...")
    records = []
    for _, row in products_df.iterrows():
        lo, hi = LEAD_TIME_RANGE.get(row["category"], (7, 14))
        records.append({
            "product_id":     row["product_id"],
            "lead_time_days": int(np.random.randint(lo, hi + 1)),
            "min_order_qty":  int(np.random.choice(MIN_ORDER_OPTIONS.get(row["category"], [10]))),
        })
    suppliers = pd.DataFrame(records)
    print(f"  ✓ {len(suppliers)} supplier records")
    return suppliers


def transform_sales_context(df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
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
    merged = sales_df.merge(context, on=["product_id", "date"], how="left")
    merged["is_holiday_promo"] = merged["is_holiday_promo"].astype(bool)
    merged["price_at_sale"] = merged["price_at_sale"].round(2)
    merged["discount_pct"] = merged["discount_pct"].round(2)
    merged["competitor_pricing"] = merged["competitor_pricing"].round(2)
    return merged[["sale_id", "weather_condition", "is_holiday_promo",
                   "price_at_sale", "discount_pct", "competitor_pricing", "seasonality"]]


def load(products, sales, inventory, suppliers) -> pd.DataFrame:
    print("▶ Loading ke database...")
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        print("  Truncating existing data...")
        conn.execute(text(
            "TRUNCATE TABLE sales_context, sales, inventory, suppliers, products RESTART IDENTITY CASCADE"
        ))
        products.to_sql("products", conn, if_exists="append", index=False)
        print(f"  ✓ {len(products)} products loaded")

        sales.to_sql("sales", conn, if_exists="append", index=False)
        print(f"  ✓ {len(sales):,} sales loaded")

        inventory.to_sql("inventory", conn, if_exists="append", index=False)
        print(f"  ✓ {len(inventory)} inventory records loaded")

        suppliers.to_sql("suppliers", conn, if_exists="append", index=False)
        print(f"  ✓ {len(suppliers)} supplier records loaded")

    with engine.connect() as conn:
        sales_with_id = pd.read_sql(
            "SELECT sale_id, product_id, date FROM sales ORDER BY sale_id",
            conn,
            parse_dates=["date"]
        )

    print(f"  ✓ Fetched {len(sales_with_id):,} sale_ids")
    return sales_with_id


def load_context(context_df: pd.DataFrame) -> None:
    print("▶ Loading sales_context...")
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        context_df.to_sql("sales_context", conn, if_exists="append", index=False)
    print(f"  ✓ {len(context_df):,} context records loaded")


def verify() -> None:
    print("\n▶ Verifying...")
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        for table in ["products", "sales", "inventory", "suppliers", "sales_context"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:<20} {count:>8,} rows")


def run():
    print("=" * 50)
    print("ETL Pipeline — Procurement Agent")
    print("=" * 50)

    raw_df       = extract()
    products_df  = transform_products(raw_df)
    sales_df     = transform_sales(raw_df)
    inventory_df = transform_inventory(raw_df)
    suppliers_df = transform_suppliers(products_df)

    sales_with_id = load(products_df, sales_df, inventory_df, suppliers_df)

    context_df = transform_sales_context(raw_df, sales_with_id)
    load_context(context_df)

    verify()

    print("\n✓ ETL selesai.")


if __name__ == "__main__":
    run()
