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
            "Run scripts/download_headlines.py first."
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
