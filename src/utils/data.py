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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_numeric_features(
    ticker: str = "AAPL",
    split: str = "train",
) -> pd.DataFrame:
    """Load z-normalized numeric features for the Executor agent.

    Args:
        ticker: Stock ticker symbol.
        split: One of "train", "val", "test".

    Returns:
        DataFrame with DatetimeIndex and ~15 numeric feature columns.
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T1)")


def load_headlines(
    ticker: str = "AAPL",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Load text headlines for the Analyst agent.

    Args:
        ticker: Stock ticker symbol.
        start_date: ISO format start date (inclusive).
        end_date: ISO format end date (inclusive).

    Returns:
        List of dicts with keys: date, ticker, headline, source.
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T2)")


def get_feature_names() -> list[str]:
    """Return the ordered list of numeric feature column names.

    Returns:
        List of feature name strings matching columns in load_numeric_features().
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T1)")


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
