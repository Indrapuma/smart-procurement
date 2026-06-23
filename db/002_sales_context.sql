CREATE TABLE IF NOT EXISTS sales_context (
    sale_id             INTEGER PRIMARY KEY REFERENCES sales(sale_id),
    weather_condition   VARCHAR(50),
    is_holiday_promo    BOOLEAN NOT NULL DEFAULT FALSE,
    price_at_sale       NUMERIC(10,2),
    discount_pct        NUMERIC(5,2),
    competitor_pricing  NUMERIC(10,2),
    seasonality         VARCHAR(20)
);