# Phase 1: Data Pipeline

**Project:** Robust Trinity — Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Timeline:** Week 2, ~8-12 hours total
**Goal:** Download and process all data needed for both the Executor (numeric market features) and the Analyst (text headlines), build data loading utilities, and verify the full pipeline.

**Prerequisites:** Phase 0 must be complete. The project structure exists at `/Users/shivamgera/projects/research1`, all dependencies are installed, seed management and logging utilities exist in `src/utils/`.

---

## P1-T1: Download and Process Numeric Market Data

**Estimated time:** ~2.5 hours
**Dependencies:** P0-T1 (project structure), P0-T2 (seed utilities and data stubs)

### Context

The Robust Trinity system has two independent data channels — this is a core architectural invariant called **channel independence**:

1. **Executor channel** (this task): Numeric OHLCV-derived features only. No text, no sentiment, no NLP features. The Executor is a PPO-based reinforcement learning agent that sees only market microstructure data.
2. **Analyst channel** (P1-T2): Text headlines only. No numeric features.

The Executor will be trained on these features inside a custom Gymnasium trading environment (Phase 2). The features must be z-normalized so that the PPO policy network receives well-scaled inputs.

**Date ranges:**
- Training warmup: Jan 2022 – Dec 2022 (needed for rolling indicators to stabilize)
- Training: Jan 2023 – Jun 2024
- Test: Jul 2024 – Dec 2024

We use a single ticker (**AAPL**) to keep the scope manageable for the thesis. The approach generalizes to other tickers.

The project root is `/Users/shivamgera/projects/research1`. The data stubs in `src/utils/data.py` currently raise `NotImplementedError` — this task replaces them with real implementations.

### Objective

Download AAPL daily OHLCV data, engineer ~15 technical features, apply rolling z-score normalization, save as Parquet, define train/test splits, and implement the `load_numeric_features()` and `get_feature_names()` functions.

### Detailed Instructions

**Step 1: Create `src/utils/features.py` — Feature engineering module**

```python
"""Technical feature engineering for the Executor agent.

All features are derived purely from OHLCV price/volume data.
NO text-derived features are permitted (channel independence invariant).
"""

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns: log(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    RSI = 100 - 100 / (1 + RS)
    RS = average_gain / average_loss over `period` days.
    Uses exponential moving average (Wilder's smoothing).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands and %B indicator.

    Returns:
        Tuple of (upper_band, lower_band, percent_b).
        percent_b = (close - lower) / (upper - lower)
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    percent_b = (close - lower) / (upper - lower)
    return upper, lower, percent_b


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = EMA(TR, period)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    return atr


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute volume ratio: volume / SMA(volume, period)."""
    sma_vol = volume.rolling(window=period).mean()
    return volume / sma_vol


def compute_realized_volatility(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling realized volatility (std of log returns)."""
    return log_returns.rolling(window=window).std()


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: Open, High, Low, Close, Volume
            and a DatetimeIndex.

    Returns:
        DataFrame with ~14 feature columns, same index as input.
        NaN rows at the start (due to rolling windows) are NOT dropped —
        the caller should handle this.
    """
    features = pd.DataFrame(index=df.index)

    # Log returns
    features["log_return"] = compute_log_returns(df["Close"])

    # RSI
    features["rsi"] = compute_rsi(df["Close"], period=14)

    # MACD
    macd_line, signal_line, histogram = compute_macd(df["Close"])
    features["macd_line"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_histogram"] = histogram

    # Bollinger Bands
    bb_upper, bb_lower, bb_pctb = compute_bollinger_bands(df["Close"])
    features["bb_upper"] = bb_upper
    features["bb_lower"] = bb_lower
    features["bb_percent_b"] = bb_pctb

    # ATR
    features["atr"] = compute_atr(df["High"], df["Low"], df["Close"])

    # Volume ratio
    features["volume_ratio"] = compute_volume_ratio(df["Volume"])

    # Realized volatility
    features["realized_vol"] = compute_realized_volatility(features["log_return"])

    # Additional raw features useful for the agent
    features["close"] = df["Close"]  # Will be z-normalized, not raw price
    features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    features["close_open_return"] = (df["Close"] - df["Open"]) / df["Open"]

    return features


def rolling_zscore_normalize(
    df: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Apply rolling z-score normalization to all columns.

    z_t = (x_t - mean(x_{t-window:t})) / std(x_{t-window:t})

    Uses a rolling window of `window` trading days.
    For early rows where fewer than `window` days are available,
    uses all available history (min_periods=20).

    Args:
        df: Raw feature DataFrame.
        window: Rolling window size in trading days.

    Returns:
        Z-normalized DataFrame with same shape and index.
    """
    rolling_mean = df.rolling(window=window, min_periods=20).mean()
    rolling_std = df.rolling(window=window, min_periods=20).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 1e-8)
    normalized = (df - rolling_mean) / rolling_std
    return normalized


FEATURE_NAMES = [
    "log_return",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "atr",
    "volume_ratio",
    "realized_vol",
    "close",
    "high_low_range",
    "close_open_return",
]
```

That's 14 features. If you want exactly 15, add one more (e.g., `open_high_range` or `sma_20_ratio`). The exact count can be adjusted.

**Step 2: Create the data download script `scripts/download_data.py`**

```python
"""Download AAPL OHLCV data and compute features."""

import yfinance as yf
import yaml
from pathlib import Path

from src.utils.features import build_feature_dataframe, rolling_zscore_normalize, FEATURE_NAMES
from src.utils.seed import set_global_seed


def main():
    set_global_seed(42)

    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ticker = config["data"]["ticker"]
    # Download with buffer for rolling windows
    # We need data from ~2021-01-01 to allow rolling indicators to stabilize by 2022-01-01
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start="2021-01-01", end="2024-12-31", auto_adjust=True)

    print(f"Downloaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Build features
    print("Computing features...")
    features = build_feature_dataframe(df)

    # Apply rolling z-score normalization
    print("Applying rolling z-score normalization...")
    features_norm = rolling_zscore_normalize(features, window=252)

    # Drop NaN rows (from rolling windows)
    features_norm = features_norm.dropna()

    print(f"Final feature matrix: {features_norm.shape}")
    print(f"Date range: {features_norm.index[0]} to {features_norm.index[-1]}")
    print(f"Features: {list(features_norm.columns)}")

    # Save
    output_path = project_root / "data" / "processed" / "aapl_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_norm.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # Also save raw OHLCV for reference
    raw_path = project_root / "data" / "raw" / "aapl_ohlcv.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path)
    print(f"Raw OHLCV saved to {raw_path}")

    # Create split config
    # Training warmup: 2022-01-01 to 2022-12-31 (for indicators to stabilize)
    # Training: 2023-01-01 to 2024-06-30
    # Test: 2024-07-01 to 2024-12-31
    splits = {
        "warmup": {"start": "2022-01-01", "end": "2022-12-31"},
        "train": {"start": "2023-01-01", "end": "2024-06-30"},
        "test": {"start": "2024-07-01", "end": "2024-12-31"},
    }

    splits_path = project_root / "configs" / "data_splits.yaml"
    with open(splits_path, "w") as f:
        yaml.dump(splits, f, default_flow_style=False)
    print(f"Splits config saved to {splits_path}")

    # Print split sizes
    for split_name, dates in splits.items():
        mask = (features_norm.index >= dates["start"]) & (features_norm.index <= dates["end"])
        n = mask.sum()
        print(f"  {split_name}: {n} trading days")


if __name__ == "__main__":
    main()
```

**Step 3: Update `src/utils/data.py` — Replace stubs with real implementations**

Replace the existing stub functions in `src/utils/data.py` with working implementations:

```python
"""Data loading and feature engineering utilities.

This module handles loading numeric market data and text headlines
for the Robust Trinity system. The Analyst and Executor agents use
completely independent data channels:
- Executor: numeric OHLCV-derived features (z-normalized)
- Analyst: text headlines (no numeric features)

This channel independence is a core architectural invariant.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.features import FEATURE_NAMES

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS = PROJECT_ROOT / "configs"


def _load_splits() -> dict:
    """Load train/test split date ranges from config."""
    splits_path = CONFIGS / "data_splits.yaml"
    with open(splits_path) as f:
        return yaml.safe_load(f)


def load_numeric_features(
    ticker: str = "AAPL",
    split: str = "train",
) -> pd.DataFrame:
    """Load z-normalized numeric features for the Executor agent.

    Args:
        ticker: Stock ticker symbol (currently only 'AAPL' supported).
        split: One of "train", "test", "warmup", or "all".

    Returns:
        DataFrame with DatetimeIndex and 14 numeric feature columns,
        all z-score normalized.
    """
    parquet_path = DATA_PROCESSED / f"{ticker.lower()}_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {parquet_path}. "
            "Run scripts/download_data.py first."
        )

    df = pd.read_parquet(parquet_path)

    if split == "all":
        return df

    splits = _load_splits()
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Available: {list(splits.keys())}")

    date_range = splits[split]
    mask = (df.index >= date_range["start"]) & (df.index <= date_range["end"])
    return df.loc[mask]


def load_raw_ohlcv(ticker: str = "AAPL") -> pd.DataFrame:
    """Load raw OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex.
    """
    path = DATA_RAW / f"{ticker.lower()}_ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw OHLCV not found: {path}")
    return pd.read_parquet(path)


def load_headlines(
    ticker: str = "AAPL",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Load text headlines for the Analyst agent.

    Args:
        ticker: Stock ticker symbol.
        start_date: ISO format start date (inclusive). None = no lower bound.
        end_date: ISO format end date (inclusive). None = no upper bound.

    Returns:
        List of dicts with keys: date, ticker, headline, source.
    """
    import json

    headlines_path = DATA_PROCESSED / "headlines.json"
    if not headlines_path.exists():
        raise FileNotFoundError(
            f"Headlines file not found: {headlines_path}. "
            "Run P1-T2 data preparation first."
        )

    with open(headlines_path) as f:
        headlines = json.load(f)

    # Filter by ticker
    headlines = [h for h in headlines if h["ticker"] == ticker]

    # Filter by date range
    if start_date:
        headlines = [h for h in headlines if h["date"] >= start_date]
    if end_date:
        headlines = [h for h in headlines if h["date"] <= end_date]

    return headlines


def get_feature_names() -> list[str]:
    """Return the ordered list of numeric feature column names.

    Returns:
        List of 14 feature name strings matching columns in load_numeric_features().
    """
    return list(FEATURE_NAMES)


def verify_channel_independence(features_df: pd.DataFrame) -> None:
    """Assert that the numeric feature DataFrame contains no text-derived columns.

    This is a core architectural invariant: the Executor sees ONLY numeric
    market data, never any text or NLP features.

    Args:
        features_df: The numeric features DataFrame.

    Raises:
        AssertionError: If any column name suggests text-derived data.
    """
    text_indicators = ["sentiment", "text", "headline", "nlp", "embedding", "token"]
    for col in features_df.columns:
        for indicator in text_indicators:
            assert indicator not in col.lower(), (
                f"Channel independence violation: column '{col}' appears to be text-derived "
                f"(contains '{indicator}'). Executor must only see numeric market features."
            )
```

**Step 4: Create tests `tests/test_features.py`**

```python
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


class TestFeatureMatrix:
    def test_column_count(self):
        """Feature matrix should have expected number of columns."""
        assert len(FEATURE_NAMES) >= 14

    def test_no_text_features(self):
        """Verify channel independence — no text-derived column names."""
        from src.utils.data import verify_channel_independence
        # Create a dummy DataFrame with our feature names
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
        # Create data with known distribution
        data = pd.DataFrame({
            "a": np.random.randn(500) * 10 + 50,
            "b": np.random.randn(500) * 2 - 3,
        })
        normed = rolling_zscore_normalize(data, window=252)
        valid = normed.dropna()
        # After normalization, mean should be close to 0, std close to 1
        # (not exactly, because rolling window)
        for col in valid.columns:
            assert abs(valid[col].mean()) < 1.0, f"{col} mean too far from 0"
            assert 0.2 < valid[col].std() < 3.0, f"{col} std out of reasonable range"
```

**Step 5: Run and verify**

```bash
python scripts/download_data.py
pytest tests/test_features.py -v
```

**Step 6: Commit**

```bash
git add src/utils/features.py src/utils/data.py scripts/download_data.py tests/test_features.py configs/data_splits.yaml
git commit -m "P1-T1: download AAPL data, feature engineering, rolling z-score normalization"
```

### Acceptance Criteria

1. `data/processed/aapl_features.parquet` exists and contains a DataFrame with 14 feature columns
2. `data/raw/aapl_ohlcv.parquet` exists with raw OHLCV data
3. `configs/data_splits.yaml` exists with warmup/train/test date ranges
4. `load_numeric_features("AAPL", "train")` returns a DataFrame with ~375 rows (trading days Jan 2023 – Jun 2024)
5. `load_numeric_features("AAPL", "test")` returns a DataFrame with ~125 rows (trading days Jul – Dec 2024)
6. All features are z-normalized (no raw prices in the processed output)
7. `verify_channel_independence()` passes on the feature DataFrame
8. `pytest tests/test_features.py -v` passes all tests
9. No NaN values in the processed feature DataFrame

### Files to Create

- `src/utils/features.py`
- `scripts/download_data.py`
- `tests/test_features.py`
- `configs/data_splits.yaml` (generated by script)
- `data/processed/aapl_features.parquet` (generated by script)
- `data/raw/aapl_ohlcv.parquet` (generated by script)

### Files to Modify

- `src/utils/data.py` (replace stubs with real implementations)

### Human Checkpoint

- Run `python scripts/download_data.py` and verify it completes
- Check that the date ranges in `configs/data_splits.yaml` are correct
- Verify the number of trading days per split is reasonable (~250/year)
- Spot-check a few feature values to ensure they make sense

---

## P1-T2: Source and Prepare Text Data for Analyst

**Estimated time:** ~2.5 hours
**Dependencies:** P0-T1 (project structure), P0-T2 (utility stubs). Can run in parallel with P1-T1.

### Context

The Robust Trinity system has an **Analyst agent** that is LLM-based (Claude in production, Llama 3.2 via Ollama for development). The Analyst processes financial text headlines and outputs a trading decision tuple `(d, r)` where:
- `d ∈ {hold, buy, sell}` — the trading decision
- `r` — free-text reasoning

**Channel independence** is a core architectural invariant: the Analyst sees ONLY text, NEVER numeric features. The Executor sees ONLY numeric features, NEVER text. They are fused only at the Consistency Gate (C-Gate) which compares their output distributions.

For this task, we need two types of text data:

1. **FinancialPhraseBank** (Malo et al. 2014): A standard dataset of 4,845 financial sentences with sentiment labels (positive/negative/neutral). This is used to VALIDATE the Analyst's sentiment understanding, not to train it (the LLM is pre-trained).

2. **Real headline dataset aligned with AAPL price data**: For each trading day in our date range (Jan 2022 – Dec 2024), we need at least one real financial news headline that the Analyst can process during simulation. We use the **Polygon.io REST API** (free "Basic" tier) to fetch historical AAPL-tagged news articles. The API returns real headlines from publishers like Benzinga, Investing.com, MarketWatch, and others — complete with ticker tags and publication timestamps.

**Why real headlines are mandatory.** An earlier draft of this task used templated/curated headlines selected based on the day's actual return (e.g., bullish template on +1% days). This creates **data leakage**: the headline perfectly encodes the day's price direction, giving the Analyst an unfair information advantage that would not exist in production. Real headlines avoid this problem — they reflect journalist judgment and publicly available information at the time of publication, not future price movements.

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Download FinancialPhraseBank, fetch real AAPL-tagged headlines from Polygon.io aligned with trading dates, store both in the correct locations, and verify data quality.

### Detailed Instructions

**Step 1: Download FinancialPhraseBank**

The FinancialPhraseBank dataset is available from multiple sources:
- Kaggle: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- Hugging Face: `financial_phrasebank` dataset
- Direct: https://www.researchgate.net/publication/251231364

The most reliable approach is to use the Hugging Face `datasets` library:

```python
# scripts/download_phrasebank.py
"""Download FinancialPhraseBank dataset."""

from pathlib import Path
import csv

try:
    from datasets import load_dataset
    ds = load_dataset("financial_phrasebank", "sentences_allagree")
    data = ds["train"]
except Exception:
    # Fallback: try alternative source
    print("Hugging Face datasets not available, trying alternative...")
    # If datasets library is not installed, add it to requirements or
    # download manually from Kaggle
    raise

project_root = Path(__file__).parent.parent
output_path = project_root / "data" / "raw" / "financial_phrasebank.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence", "label"])
    for item in data:
        # Labels: 0=negative, 1=neutral, 2=positive
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        writer.writerow([item["sentence"], label_map[item["label"]]])

print(f"Saved {len(data)} sentences to {output_path}")
```

If the `datasets` library is not already in `pyproject.toml`, add it temporarily or download manually. The key thing is to get a CSV with columns `sentence, label` where label ∈ {positive, negative, neutral}.

**Step 2: Fetch real AAPL headlines from Polygon.io**

Prerequisites:
- Sign up for a free Polygon.io account at https://polygon.io (the "Basic" tier is free, no credit card required).
- Copy your API key and add it to `.env` as `POLYGON_API_KEY=pk_...`.
- The free tier allows 5 API requests per minute. The script respects this with a 12-second sleep between paginated requests.

Create `scripts/download_headlines.py`:

```python
"""Fetch real AAPL-tagged headlines from Polygon.io REST API.

Uses the /v2/reference/news endpoint with ticker=AAPL to retrieve
historical news articles. Paginates through the full date range
(Jan 2022 – Dec 2024), deduplicates to one headline per trading day,
and saves as JSON.

API docs: https://polygon.io/docs/rest/stocks/news
Free tier: 5 requests/minute, no daily cap, historical access.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
output_path = project_root / "data" / "processed" / "headlines.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Polygon.io configuration
BASE_URL = "https://api.polygon.io/v2/reference/news"
TICKER = "AAPL"
DATE_START = "2022-01-01"  # warmup period start
DATE_END = "2024-12-31"    # test period end
PAGE_LIMIT = 1000          # max allowed per request
RATE_LIMIT_SLEEP = 12.5    # seconds between requests (5 req/min = 12s)


def fetch_all_articles(api_key: str) -> list[dict]:
    """Fetch all AAPL-tagged articles from Polygon.io, paginating via next_url.

    Returns raw article dicts from the API with keys like 'title',
    'published_utc', 'publisher', 'tickers', etc.
    """
    articles: list[dict] = []
    url = BASE_URL
    params = {
        "ticker": TICKER,
        "published_utc.gte": f"{DATE_START}T00:00:00Z",
        "published_utc.lte": f"{DATE_END}T23:59:59Z",
        "limit": PAGE_LIMIT,
        "sort": "published_utc",
        "order": "asc",
        "apiKey": api_key,
    }

    page = 0
    while url:
        page += 1
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Request failed on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        articles.extend(results)
        print(f"  Page {page}: fetched {len(results)} articles "
              f"(total: {len(articles)})")

        # Pagination: Polygon returns a next_url with a cursor token.
        # On subsequent requests, use next_url directly (no params needed).
        next_url = data.get("next_url")
        if next_url:
            # Append API key to next_url (Polygon requires it on every request)
            url = f"{next_url}&apiKey={api_key}"
            params = {}  # next_url already contains all query params
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            url = None

    return articles


def load_trading_dates(ticker: str) -> set[str]:
    """Load the set of actual trading dates from the raw OHLCV file.

    These are dates where the market was open and AAPL traded.
    Used to filter headlines to only trading days.
    """
    ohlcv_path = project_root / "data" / "raw" / f"{ticker.lower()}_ohlcv.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(
            f"Raw OHLCV not found at {ohlcv_path}. Run P1-T1 first."
        )
    df = pd.read_parquet(ohlcv_path)
    return {d.strftime("%Y-%m-%d") for d in df.index}


def deduplicate_to_one_per_day(
    articles: list[dict],
    trading_dates: set[str],
) -> list[dict]:
    """Select one headline per trading day from the fetched articles.

    For each trading day, picks the first article (by published_utc)
    that mentions AAPL in its tickers list. Articles on non-trading
    days (weekends, holidays) are discarded.

    Returns headline dicts in the output schema:
        {date, ticker, headline, source}
    """
    # Group articles by trading date
    day_to_articles: dict[str, list[dict]] = {}
    for article in articles:
        pub = article.get("published_utc", "")
        if not pub:
            continue
        date_str = pub[:10]  # "YYYY-MM-DD"

        if date_str not in trading_dates:
            continue

        # Confirm AAPL is actually tagged (not just a passing mention)
        tickers = article.get("tickers", [])
        if TICKER not in tickers:
            continue

        if date_str not in day_to_articles:
            day_to_articles[date_str] = []
        day_to_articles[date_str].append(article)

    # For each trading day, take the first article (already sorted by
    # published_utc ascending from the API)
    headlines = []
    for date_str in sorted(day_to_articles.keys()):
        article = day_to_articles[date_str][0]
        publisher = article.get("publisher", {})
        headlines.append({
            "date": date_str,
            "ticker": TICKER,
            "headline": article.get("title", "").strip(),
            "source": publisher.get("name", "unknown"),
        })

    return headlines


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set in environment.")
        print("Sign up for a free account at https://polygon.io")
        print("Then add POLYGON_API_KEY=pk_... to your .env file.")
        sys.exit(1)

    print(f"Fetching {TICKER} headlines from Polygon.io...")
    print(f"Date range: {DATE_START} to {DATE_END}")
    print(f"Rate limit: {RATE_LIMIT_SLEEP}s between pages\n")

    # Step 1: Fetch all articles from the API
    articles = fetch_all_articles(api_key)
    print(f"\nTotal articles fetched: {len(articles)}")

    if not articles:
        print("ERROR: No articles returned. Check your API key and try again.")
        sys.exit(1)

    # Step 2: Load trading calendar from P1-T1 output
    trading_dates = load_trading_dates(TICKER)
    print(f"Trading dates in OHLCV: {len(trading_dates)}")

    # Step 3: Deduplicate to one headline per trading day
    headlines = deduplicate_to_one_per_day(articles, trading_dates)
    print(f"Headlines after deduplication: {len(headlines)}")

    # Step 4: Report coverage
    covered_dates = {h["date"] for h in headlines}
    # Only count trading dates within our target range
    target_dates = {
        d for d in trading_dates
        if DATE_START <= d <= DATE_END
    }
    coverage = len(covered_dates & target_dates) / len(target_dates) if target_dates else 0
    missing = sorted(target_dates - covered_dates)

    print(f"Coverage: {coverage:.1%} of {len(target_dates)} trading days")
    if missing:
        print(f"Missing dates ({len(missing)}):")
        for d in missing[:20]:
            print(f"  {d}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    # Step 5: Save
    with open(output_path, "w") as f:
        json.dump(headlines, f, indent=2)

    print(f"\nSaved {len(headlines)} headlines to {output_path}")
    print(f"Date range: {headlines[0]['date']} to {headlines[-1]['date']}")

    # Summary statistics
    sources = {}
    for h in headlines:
        sources[h["source"]] = sources.get(h["source"], 0) + 1
    print("\nHeadlines by source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {count}")

    # Warn if coverage is low
    if coverage < 0.90:
        print(f"\nWARNING: Coverage is {coverage:.1%}, below the 90% target.")
        print("The C-Gate handles missing Analyst signals by treating them as")
        print("conflict (delta=1.0, action=flat), but low coverage degrades")
        print("the system's ability to demonstrate agreement regimes.")


if __name__ == "__main__":
    main()
```

**Step 3: Run both download scripts**

```bash
# Ensure P1-T1 has been run first (need raw OHLCV data for trading date alignment)
# Ensure POLYGON_API_KEY is set in .env
python scripts/download_phrasebank.py
python scripts/download_headlines.py
```

The headline download will take several minutes due to rate limiting (12.5 seconds per page, typically 3-8 pages depending on news volume for AAPL). Progress is printed to stdout.

**Step 4: Verify data quality**

Check:
- `data/raw/financial_phrasebank.csv` has ~2,000-5,000 rows (varies by agreement subset)
- `data/processed/headlines.json` has 400+ entries (one per trading day, coverage may not be 100%)
- Headlines are real financial news (diverse publishers, specific details, not templated)
- Headlines dates align with actual AAPL trading days (no weekends/holidays)
- Coverage report shows ≥90% of trading days have headlines
- No data leakage: headline content is NOT derived from or conditioned on that day's price return

**Step 5: Commit**

```bash
git add scripts/download_phrasebank.py scripts/download_headlines.py
git commit -m "P1-T2: source FinancialPhraseBank and fetch real AAPL headlines from Polygon.io"
```

### Acceptance Criteria

1. `data/raw/financial_phrasebank.csv` exists with `sentence` and `label` columns
2. `data/processed/headlines.json` exists as a valid JSON array of objects
3. Each headline object has keys: `date`, `ticker`, `headline`, `source`
4. Headlines cover the full date range (2022-01-01 to 2024-12-31) with ≥90% trading-day coverage
5. Headlines come from real financial news publishers (not templated or synthetic)
6. No data leakage — headline selection is NOT conditioned on that day's price return
7. No duplicate dates (exactly one headline per trading day)
8. `load_headlines("AAPL", "2023-01-01", "2023-12-31")` returns ≥200 items
9. Headline text is non-trivial (length > 10 characters, contains real content)
10. Source field contains real publisher names (e.g., "Benzinga", "Investing.com")

### Files to Create

- `scripts/download_phrasebank.py`
- `scripts/download_headlines.py`
- `data/raw/financial_phrasebank.csv` (generated)
- `data/processed/headlines.json` (generated)

### Files to Modify

- `pyproject.toml` — add `requests>=2.31.0` to dependencies (needed for Polygon.io API calls)
- `.env.example` — add `POLYGON_API_KEY=`

### Human Checkpoint

- **CRITICAL:** Review at least 20 headlines to verify they are real financial news articles, not synthetic or templated
- Spot-check 5-10 headlines against external sources (Google the headline text) to confirm authenticity
- Verify headline diversity — no repeated templates, varied publishers, specific details
- Check the coverage report — if below 90%, decide whether to supplement missing days or accept the gaps (the C-Gate already handles missing Analyst signals as conflict)
- Check the FinancialPhraseBank for expected format (sentence + label)
- Verify no data leakage: skim 10 bullish-sounding headlines and confirm they do NOT systematically appear on positive-return days only

---

## P1-T3: Create Data Loading Utilities and Verify Pipeline

**Estimated time:** ~2 hours
**Dependencies:** P1-T1 (numeric features must exist), P1-T2 (headlines must exist)

### Context

The Robust Trinity system has two independent data channels:

1. **Executor channel**: Numeric OHLCV-derived features loaded from `data/processed/aapl_features.parquet`. These are z-normalized, with ~14 features per trading day.
2. **Analyst channel**: Text headlines loaded from `data/processed/headlines.json`. One headline per trading day.

Both were created in P1-T1 and P1-T2. The data loading functions in `src/utils/data.py` have been partially implemented in P1-T1 (replacing the original stubs from P0-T2). This task finalizes the loading utilities, creates comprehensive integration tests, and builds an exploration notebook.

**Channel independence** is a core invariant: the numeric features DataFrame must NEVER contain text-derived columns (sentiment, NLP features, etc.). This is enforced by `verify_channel_independence()`.

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Finalize all data loading utilities, create comprehensive integration tests, and build a data exploration notebook that visualizes features and headlines.

### Detailed Instructions

**Step 1: Create comprehensive integration tests `tests/test_data_pipeline.py`**

```python
"""Integration tests for the complete data pipeline.

These tests verify that:
1. Numeric features load correctly with proper shapes and types
2. Headlines load correctly with proper structure
3. Channel independence is maintained
4. Date alignment between channels is correct
5. No NaN values in processed features
"""

import pytest
import numpy as np
import pandas as pd

from src.utils.data import (
    load_numeric_features,
    load_headlines,
    load_raw_ohlcv,
    get_feature_names,
    verify_channel_independence,
)


class TestNumericFeatures:
    """Test numeric feature loading for the Executor agent."""

    def test_train_split_loads(self):
        df = load_numeric_features("AAPL", split="train")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 200  # ~375 trading days expected
        assert len(df) < 500

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
            assert required_keys.issubset(h.keys()), f"Missing keys: {required_keys - h.keys()}"

    def test_date_filtering(self):
        headlines = load_headlines("AAPL", start_date="2023-06-01", end_date="2023-06-30")
        assert len(headlines) > 10  # ~20 trading days in June, expect ≥90% coverage
        assert len(headlines) < 30
        for h in headlines:
            assert h["date"] >= "2023-06-01"
            assert h["date"] <= "2023-06-30"

    def test_headlines_nonempty(self):
        headlines = load_headlines("AAPL")
        for h in headlines:
            assert len(h["headline"]) > 10, f"Headline too short: {h['headline']}"


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

    def test_invalid_ticker(self):
        with pytest.raises(FileNotFoundError):
            load_raw_ohlcv("NONEXISTENT_TICKER_XYZ")
```

**Step 2: Create data exploration notebook `notebooks/01_data_exploration.ipynb`**

Since creating a notebook programmatically is verbose, create it as a Python script that can be run or converted:

```python
# notebooks/01_data_exploration.py
# Convert to notebook: jupyter nbconvert --to notebook --execute 01_data_exploration.py
"""
# Data Exploration — Robust Trinity
## Verify numeric features, headlines, and channel independence
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import (
    load_numeric_features,
    load_headlines,
    load_raw_ohlcv,
    get_feature_names,
    verify_channel_independence,
)

# %% Load data
print("=" * 60)
print("NUMERIC FEATURES")
print("=" * 60)
features_all = load_numeric_features("AAPL", split="all")
features_train = load_numeric_features("AAPL", split="train")
features_test = load_numeric_features("AAPL", split="test")
raw = load_raw_ohlcv("AAPL")

print(f"All features shape: {features_all.shape}")
print(f"Train shape: {features_train.shape}")
print(f"Test shape: {features_test.shape}")
print(f"Feature names: {get_feature_names()}")
print(f"\nTrain date range: {features_train.index[0]} to {features_train.index[-1]}")
print(f"Test date range: {features_test.index[0]} to {features_test.index[-1]}")

# Check NaN
nan_counts = features_all.isna().sum()
print(f"\nNaN counts per column:\n{nan_counts}")
assert nan_counts.sum() == 0, "Found NaN values!"

# Channel independence
verify_channel_independence(features_all)
print("\nChannel independence: VERIFIED")

# %% Plot raw price
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(raw.index, raw["Close"], linewidth=0.8)
axes[0].axvline(pd.Timestamp("2023-01-01"), color="green", linestyle="--", label="Train start")
axes[0].axvline(pd.Timestamp("2024-07-01"), color="red", linestyle="--", label="Test start")
axes[0].set_title("AAPL Close Price")
axes[0].set_ylabel("Price ($)")
axes[0].legend()

axes[1].bar(raw.index, raw["Volume"], width=1, alpha=0.5)
axes[1].set_title("AAPL Volume")
axes[1].set_ylabel("Volume")
plt.tight_layout()
plt.savefig("notebooks/price_and_volume.png", dpi=150)
plt.close()

# %% Feature distributions
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
feature_names = get_feature_names()
for i, (ax, feat) in enumerate(zip(axes.flat, feature_names)):
    ax.hist(features_train[feat].values, bins=50, alpha=0.7, density=True)
    ax.set_title(feat, fontsize=9)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    mean = features_train[feat].mean()
    std = features_train[feat].std()
    ax.set_xlabel(f"μ={mean:.2f}, σ={std:.2f}", fontsize=7)

# Hide unused subplots
for j in range(len(feature_names), len(axes.flat)):
    axes.flat[j].set_visible(False)

plt.suptitle("Feature Distributions (Train Set, Z-Normalized)", fontsize=14)
plt.tight_layout()
plt.savefig("notebooks/feature_distributions.png", dpi=150)
plt.close()

# %% Feature correlations
corr = features_train.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_title("Feature Correlation Matrix (Train Set)")
plt.tight_layout()
plt.savefig("notebooks/feature_correlations.png", dpi=150)
plt.close()

# %% Headlines
print("\n" + "=" * 60)
print("HEADLINES")
print("=" * 60)
headlines = load_headlines("AAPL")
print(f"Total headlines: {len(headlines)}")
print(f"Date range: {headlines[0]['date']} to {headlines[-1]['date']}")

# Sample headlines
print("\nSample headlines:")
for h in headlines[:5]:
    print(f"  [{h['date']}] {h['headline']}")

print("\nDone! Check notebooks/ directory for plots.")
```

**Step 3: Run tests and exploration**

```bash
pytest tests/test_data_pipeline.py -v
python notebooks/01_data_exploration.py
```

**Step 4: Commit**

```bash
git add tests/test_data_pipeline.py notebooks/01_data_exploration.py
git commit -m "P1-T3: integration tests and data exploration for full pipeline"
```

### Acceptance Criteria

1. `pytest tests/test_data_pipeline.py -v` passes ALL tests (14+ tests)
2. `notebooks/01_data_exploration.py` runs without errors and generates plots
3. Feature distributions are approximately standard normal (visible in histograms)
4. No NaN values anywhere in the processed feature matrix
5. Train/test sets have zero date overlap
6. Headlines cover >90% of trading days in the training period
7. Channel independence assertion passes
8. Feature correlation matrix shows no unexpected perfect correlations (which would indicate redundant features)
9. Plots saved to `notebooks/` directory: `price_and_volume.png`, `feature_distributions.png`, `feature_correlations.png`

### Files to Create

- `tests/test_data_pipeline.py`
- `notebooks/01_data_exploration.py`
- `notebooks/price_and_volume.png` (generated)
- `notebooks/feature_distributions.png` (generated)
- `notebooks/feature_correlations.png` (generated)

### Files to Modify

- None (all loading utilities were finalized in P1-T1)

### Human Checkpoint

- **Review the feature distribution plots**: all features should be roughly bell-shaped and centered near zero
- **Review the correlation matrix**: flag any features with |correlation| > 0.95 (consider dropping one)
- **Read 10+ sample headlines**: verify they are plausible financial headlines
- **Check train/test split**: ensure no data leakage
- **Approve the pipeline** before proceeding to Phase 2 (Executor training)
