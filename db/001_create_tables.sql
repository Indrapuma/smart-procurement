CREATE TABLE products (
    product_id   VARCHAR(50) PRIMARY KEY,
    name         VARCHAR(200) NOT NULL,
    category     VARCHAR(100),
    unit_price   NUMERIC(10,2)   
);

CREATE TABLE sales (
    sale_id     SERIAL PRIMARY KEY,
    product_id  VARCHAR(50) REFERENCES products(product_id),
    date        DATE NOT NULL,
    qty_sold    INTEGER NOT NULL
);

CREATE TABLE inventory (
    product_id    VARCHAR(50) PRIMARY KEY REFERENCES products(product_id),
    current_stock INTEGER NOT NULL,
    last_updated  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE suppliers (
    supplier_id    SERIAL PRIMARY KEY,
    product_id     VARCHAR(50) REFERENCES products(product_id),
    lead_time_days INTEGER NOT NULL,
    min_order_qty  INTEGER DEFAULT 1
);


CREATE INDEX idx_sales_product_date ON sales(product_id, date);