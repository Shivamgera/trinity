"""Tests for feature engineering and data loading."""

import numpy as np
import pandas as pd
import pytest

from src.utils.features import (
    compute_log_returns,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    build_feature_dataframe,
    rolling_zscore_normalize,
    FEATURE_NAMES,
)


class TestLogReturns:
    def test_basic(self):
        close = pd.Series([100.0, 101.0, 99.0, 102.0])
        lr = compute_log_returns(close)
        assert pd.isna(lr.iloc[0])
        assert lr.iloc[1] == pytest.approx(np.log(101 / 100), abs=1e-10)

    def test_shape_preserved(self):
        close = pd.Series(np.random.rand(100) + 50)
        lr = compute_log_returns(close)
        assert len(lr) == len(close)


class TestRSI:
    def test_range(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        rsi = compute_rsi(close)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestMACD:
    def test_returns_three_series(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        macd_line, signal_line, histogram = compute_macd(close)
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)


class TestBollingerBands:
    def test_upper_above_lower(self):
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        upper, lower, pctb = compute_bollinger_bands(close)
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()


class TestFeatureMatrix:
    def test_column_count(self):
        """Feature matrix should have expected number of columns."""
        assert len(FEATURE_NAMES) == 14

    def test_build_feature_dataframe(self):
        """Test building features from OHLCV data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": np.cumsum(np.random.randn(n)) + 100,
                "High": np.cumsum(np.random.randn(n)) + 102,
                "Low": np.cumsum(np.random.randn(n)) + 98,
                "Close": np.cumsum(np.random.randn(n)) + 100,
                "Volume": np.random.randint(1_000_000, 10_000_000, n),
            },
            index=dates,
        )
        features = build_feature_dataframe(df)
        assert list(features.columns) == FEATURE_NAMES
        assert len(features) == n

    def test_no_text_features(self):
        """Verify channel independence — no text-derived column names."""
        from src.utils.data import verify_channel_independence

        df = pd.DataFrame(
            np.random.randn(10, len(FEATURE_NAMES)),
            columns=FEATURE_NAMES,
        )
        # Should not raise
        verify_channel_independence(df)

    def test_channel_independence_violation(self):
        """Should raise if a text-derived column is present."""
        from src.utils.data import verify_channel_independence

        df = pd.DataFrame({"sentiment_score": [0.5], "close": [100.0]})
        with pytest.raises(AssertionError, match="Channel independence violation"):
            verify_channel_independence(df)


class TestZScoreNormalization:
    def test_approximately_standard(self):
        """After z-scoring, data should be roughly mean=0, std=1."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "a": np.random.randn(500) * 10 + 50,
                "b": np.random.randn(500) * 2 - 3,
            }
        )
        normed = rolling_zscore_normalize(data, window=252)
        valid = normed.dropna()
        for col in valid.columns:
            assert abs(valid[col].mean()) < 1.0, f"{col} mean too far from 0"
            assert 0.2 < valid[col].std() < 3.0, f"{col} std out of reasonable range"

    def test_shape_preserved(self):
        """Z-score normalization should preserve DataFrame shape."""
        data = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
            }
        )
        normed = rolling_zscore_normalize(data, window=50)
        assert normed.shape == data.shape
