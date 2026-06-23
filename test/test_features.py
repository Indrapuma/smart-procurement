import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from ml.features import (
    _encode_categoricals,
    _build_lag_features,
    build_feature_matrix,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_raw_df(n_days: int = 60, n_products: int = 2) -> pd.DataFrame:
    """Synthetic DataFrame yang meniru output _load_raw()."""
    records = []
    base = pd.Timestamp("2023-01-01")
    for pid in [f"P{i:04d}" for i in range(1, n_products + 1)]:
        for d in range(n_days):
            records.append({
                "sale_id":          len(records) + 1,
                "product_id":       pid,
                "date":             base + pd.Timedelta(days=d),
                "qty_sold":         int(np.random.randint(10, 100)),
                "weather_condition": np.random.choice(["Sunny", "Rainy", "Cloudy"]),
                "is_holiday_promo": bool(np.random.choice([True, False])),
                "price_at_sale":    round(float(np.random.uniform(5, 50)), 2),
                "category":         np.random.choice(["Electronics", "Groceries"]),
            })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── _encode_categoricals ──────────────────────────────────────────────────────

def test_encode_categoricals_adds_encoded_columns():
    df = _make_raw_df()
    result, _ = _encode_categoricals(df.copy())
    assert "weather_encoded" in result.columns
    assert "category_encoded" in result.columns

def test_encode_categoricals_returns_encoder_mapping():
    df = _make_raw_df()
    _, encoders = _encode_categoricals(df.copy())
    assert "weather_condition" in encoders
    assert "category" in encoders
    assert "classes" in encoders["weather_condition"]
    assert "mapping" in encoders["category"]

def test_encode_categoricals_encoded_values_are_integers():
    df = _make_raw_df()
    result, _ = _encode_categoricals(df.copy())
    assert result["weather_encoded"].dtype in (int, np.int64, np.int32)
    assert result["category_encoded"].dtype in (int, np.int64, np.int32)

def test_encode_categoricals_handles_missing_values():
    df = _make_raw_df()
    df.loc[0, "weather_condition"] = None
    result, _ = _encode_categoricals(df.copy())
    assert result["weather_encoded"].isna().sum() == 0


# ── _build_lag_features ───────────────────────────────────────────────────────

def test_build_lag_features_adds_expected_columns():
    df = _make_raw_df(n_days=60)
    result = _build_lag_features(df.copy())
    for col in ("lag_1_week", "lag_4_week", "rolling_avg_4w", "rolling_std_4w"):
        assert col in result.columns

def test_build_lag_features_lag1_equals_shifted_qty():
    df = _make_raw_df(n_days=60, n_products=1)
    df = df.sort_values("date").reset_index(drop=True)
    result = _build_lag_features(df.copy())
    result = result.sort_values("date").reset_index(drop=True)
    # baris ke-7 harus sama dengan qty_sold baris ke-0
    assert result.loc[7, "lag_1_week"] == df.loc[0, "qty_sold"]

def test_build_lag_features_no_data_leakage():
    """rolling_avg_4w tidak boleh mengikutkan hari H sendiri."""
    df = _make_raw_df(n_days=60, n_products=1)
    df = df.sort_values("date").reset_index(drop=True)
    result = _build_lag_features(df.copy())
    # ubah qty_sold hari terakhir jadi nilai ekstrem
    last_idx = result.index[-1]
    original_avg = result.loc[last_idx, "rolling_avg_4w"]
    df.loc[last_idx, "qty_sold"] = 999999
    result2 = _build_lag_features(df.copy())
    # rolling_avg_4w hari itu tidak boleh berubah karena pakai shift(1)
    assert result.loc[last_idx, "rolling_avg_4w"] == result2.loc[last_idx, "rolling_avg_4w"]

def test_build_lag_features_lag_nan_at_start():
    """Baris awal tiap produk harus NaN karena belum ada history."""
    df = _make_raw_df(n_days=60, n_products=1)
    result = _build_lag_features(df.copy())
    result = result.sort_values(["product_id", "date"]).reset_index(drop=True)
    assert pd.isna(result.loc[0, "lag_1_week"])
    assert pd.isna(result.loc[0, "lag_4_week"])


# ── build_feature_matrix ──────────────────────────────────────────────────────

FEATURE_COLS = [
    "product_id", "date",
    "month", "week_of_year",
    "is_holiday_promo", "weather_encoded", "price",
    "lag_1_week", "lag_4_week",
    "rolling_avg_4w", "rolling_std_4w",
    "category_encoded",
    "target",
]

@pytest.fixture
def mock_raw():
    return _make_raw_df(n_days=60, n_products=3)

def test_build_feature_matrix_returns_tuple(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        result = build_feature_matrix()
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_build_feature_matrix_has_all_columns(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        df, _ = build_feature_matrix()
    for col in FEATURE_COLS:
        assert col in df.columns, f"Kolom '{col}' tidak ada"

def test_build_feature_matrix_no_nulls_after_dropna(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        df, _ = build_feature_matrix()
    assert df.isnull().sum().sum() == 0

def test_build_feature_matrix_target_column_is_qty_sold(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        df, _ = build_feature_matrix()
    assert (df["target"] > 0).all()

def test_build_feature_matrix_month_range(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        df, _ = build_feature_matrix()
    assert df["month"].between(1, 12).all()

def test_build_feature_matrix_filter_by_product_id(mock_raw):
    target_id = mock_raw["product_id"].iloc[0]
    with patch("ml.features._load_raw", return_value=mock_raw):
        df, _ = build_feature_matrix(product_id=target_id)
    assert (df["product_id"] == target_id).all()

def test_build_feature_matrix_encoder_keys(mock_raw):
    with patch("ml.features._load_raw", return_value=mock_raw):
        _, encoders = build_feature_matrix()
    assert "weather_condition" in encoders
    assert "category" in encoders
