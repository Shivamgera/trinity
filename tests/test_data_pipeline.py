"""Integration tests for the complete data pipeline.

These tests verify that:
1. Numeric features load correctly with proper shapes and types
2. Headlines load correctly with proper structure
3. Channel independence is maintained
4. Date alignment between channels is correct
5. No NaN values in processed features
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.data import (
    get_feature_names,
    load_headlines,
    load_numeric_features,
    load_raw_ohlcv,
    verify_channel_independence,
)


class TestNumericFeatures:
    """Test numeric feature loading for the Executor agent."""

    def test_train_split_loads(self):
        df = load_numeric_features("AAPL", split="train")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 600  # ~753 trading days expected (2021-2023)
        assert len(df) < 900

    def test_test_split_loads(self):
        df = load_numeric_features("AAPL", split="test")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 80  # ~125 trading days expected
        assert len(df) < 200

    def test_no_nans(self):
        df = load_numeric_features("AAPL", split="all")
        assert df.isna().sum().sum() == 0, f"Found NaN values: {df.isna().sum()}"

    def test_feature_count(self):
        df = load_numeric_features("AAPL", split="train")
        expected_features = get_feature_names()
        assert len(df.columns) == len(expected_features)
        assert list(df.columns) == expected_features

    def test_z_normalized_range(self):
        """After z-normalization, most values should be in [-5, 5]."""
        df = load_numeric_features("AAPL", split="train")
        for col in df.columns:
            pct_extreme = ((df[col].abs() > 5).sum()) / len(df)
            assert pct_extreme < 0.05, (
                f"Column {col}: {pct_extreme:.1%} of values outside [-5, 5]"
            )

    def test_datetime_index(self):
        df = load_numeric_features("AAPL", split="train")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_train_test_no_overlap(self):
        train = load_numeric_features("AAPL", split="train")
        test = load_numeric_features("AAPL", split="test")
        overlap = train.index.intersection(test.index)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_channel_independence(self):
        df = load_numeric_features("AAPL", split="train")
        verify_channel_independence(df)  # Should not raise

    def test_warmup_split_loads(self):
        df = load_numeric_features("AAPL", split="warmup")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 150  # ~251 trading days expected
        assert len(df) < 300

    def test_all_split_loads(self):
        df = load_numeric_features("AAPL", split="all")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 700  # all data expected


class TestHeadlines:
    """Test headline loading for the Analyst agent."""

    def test_loads(self):
        headlines = load_headlines("AAPL")
        assert isinstance(headlines, list)
        assert len(headlines) > 400  # ~500-700 expected from Polygon.io

    def test_structure(self):
        headlines = load_headlines("AAPL")
        required_keys = {"date", "ticker", "headline", "source"}
        for h in headlines[:10]:
            assert required_keys.issubset(h.keys()), (
                f"Missing keys: {required_keys - h.keys()}"
            )

    def test_date_filtering(self):
        headlines = load_headlines("AAPL", start_date="2023-06-01", end_date="2023-06-30")
        assert len(headlines) > 10  # ~20 trading days in June
        assert len(headlines) < 30
        for h in headlines:
            assert h["date"] >= "2023-06-01"
            assert h["date"] <= "2023-06-30"

    def test_headlines_nonempty(self):
        headlines = load_headlines("AAPL")
        for h in headlines:
            assert len(h["headline"]) > 10, f"Headline too short: {h['headline']}"

    def test_no_duplicate_dates(self):
        headlines = load_headlines("AAPL")
        dates = [h["date"] for h in headlines]
        assert len(dates) == len(set(dates)), "Found duplicate dates in headlines"

    def test_real_sources(self):
        """Headlines should come from diverse real publishers."""
        headlines = load_headlines("AAPL")
        sources = set(h["source"] for h in headlines)
        assert len(sources) >= 3, f"Too few sources: {sources}"


class TestChannelAlignment:
    """Test that numeric and text data align on dates."""

    def test_date_coverage(self):
        """Headlines should cover most of the numeric data trading days."""
        features = load_numeric_features("AAPL", split="train")
        headlines = load_headlines(
            "AAPL",
            start_date=str(features.index[0].date()),
            end_date=str(features.index[-1].date()),
        )

        feature_dates = set(str(d.date()) for d in features.index)
        headline_dates = set(h["date"] for h in headlines)

        coverage = len(feature_dates & headline_dates) / len(feature_dates)
        assert coverage > 0.90, (
            f"Only {coverage:.1%} of trading days have headlines. "
            f"Missing: {len(feature_dates - headline_dates)} days"
        )


class TestRawOHLCV:
    """Test raw data loading."""

    def test_loads(self):
        df = load_raw_ohlcv("AAPL")
        assert isinstance(df, pd.DataFrame)
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        assert required_cols.issubset(set(df.columns))

    def test_row_count(self):
        df = load_raw_ohlcv("AAPL")
        assert len(df) > 900  # ~1004 rows for 2021-2024

    def test_invalid_ticker(self):
        with pytest.raises(FileNotFoundError):
            load_raw_ohlcv("NONEXISTENT_TICKER_XYZ")
