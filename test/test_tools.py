import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.sql_tools import (
    get_current_stock,
    get_sales_history,
    get_supplier_leadtime,
    get_demand_trend,
    get_seasonality_index,
    rank_products_by_risk,
)

VALID_ID  = "P0001"
INVALID_ID = "XXXXX"


# ── get_current_stock ────────────────────────────────────────────────────────

def test_get_current_stock_returns_expected_keys():
    result = get_current_stock(VALID_ID)
    assert "error" not in result
    for key in ("product_id", "name", "category", "current_stock", "last_updated"):
        assert key in result

def test_get_current_stock_product_id_matches():
    result = get_current_stock(VALID_ID)
    assert result["product_id"] == VALID_ID

def test_get_current_stock_stock_is_non_negative():
    result = get_current_stock(VALID_ID)
    assert result["current_stock"] >= 0

def test_get_current_stock_invalid_id_returns_error():
    result = get_current_stock(INVALID_ID)
    assert "error" in result


# ── get_sales_history ────────────────────────────────────────────────────────

def test_get_sales_history_returns_expected_keys():
    result = get_sales_history(VALID_ID, n_months=3)
    assert "error" not in result
    for key in ("product_id", "n_months", "records", "total_sold", "avg_daily"):
        assert key in result

def test_get_sales_history_records_not_empty():
    result = get_sales_history(VALID_ID, n_months=3)
    assert len(result["records"]) > 0

def test_get_sales_history_total_sold_positive():
    result = get_sales_history(VALID_ID, n_months=3)
    assert result["total_sold"] > 0

def test_get_sales_history_invalid_id_returns_error():
    result = get_sales_history(INVALID_ID)
    assert "error" in result


# ── get_supplier_leadtime ────────────────────────────────────────────────────

def test_get_supplier_leadtime_returns_expected_keys():
    result = get_supplier_leadtime(VALID_ID)
    assert "error" not in result
    for key in ("product_id", "lead_time_days", "min_order_qty"):
        assert key in result

def test_get_supplier_leadtime_values_positive():
    result = get_supplier_leadtime(VALID_ID)
    assert result["lead_time_days"] > 0
    assert result["min_order_qty"] > 0

def test_get_supplier_leadtime_invalid_id_returns_error():
    result = get_supplier_leadtime(INVALID_ID)
    assert "error" in result


# ── get_demand_trend ─────────────────────────────────────────────────────────

def test_get_demand_trend_returns_expected_keys():
    result = get_demand_trend(VALID_ID, n_months=3)
    assert "error" not in result
    for key in ("product_id", "n_months", "growth_rate", "avg_first_half",
                "avg_second_half", "ma7_latest"):
        assert key in result

def test_get_demand_trend_growth_rate_is_float():
    result = get_demand_trend(VALID_ID, n_months=3)
    assert isinstance(result["growth_rate"], float)

def test_get_demand_trend_invalid_id_returns_error():
    result = get_demand_trend(INVALID_ID, n_months=3)
    assert "error" in result


# ── get_seasonality_index ────────────────────────────────────────────────────

def test_get_seasonality_index_returns_expected_keys():
    result = get_seasonality_index(VALID_ID)
    assert "error" not in result
    for key in ("product_id", "overall_avg", "peak_month", "peak_index", "by_month"):
        assert key in result

def test_get_seasonality_index_by_month_has_12_entries():
    result = get_seasonality_index(VALID_ID)
    assert len(result["by_month"]) == 12

def test_get_seasonality_index_peak_index_above_zero():
    result = get_seasonality_index(VALID_ID)
    assert result["peak_index"] > 0

def test_get_seasonality_index_invalid_id_returns_error():
    result = get_seasonality_index(INVALID_ID)
    assert "error" in result


# ── rank_products_by_risk ────────────────────────────────────────────────────

VALID_RISK_VALUES = {"CRITICAL", "WARNING", "OVERSTOCK", "OK"}

def test_rank_products_by_risk_returns_list():
    result = rank_products_by_risk()
    assert isinstance(result, list)
    assert len(result) > 0

def test_rank_products_by_risk_each_item_has_required_keys():
    result = rank_products_by_risk()
    for item in result:
        for key in ("product_id", "name", "category", "current_stock",
                    "days_of_stock", "lead_time_days", "risk"):
            assert key in item

def test_rank_products_by_risk_only_valid_risk_values():
    result = rank_products_by_risk()
    for item in result:
        assert item["risk"] in VALID_RISK_VALUES

def test_rank_products_by_risk_category_filter():
    all_items = rank_products_by_risk()
    categories = {item["category"] for item in all_items}
    if categories:
        cat = next(iter(categories))
        filtered = rank_products_by_risk(category=cat)
        assert all(item["category"] == cat for item in filtered)
