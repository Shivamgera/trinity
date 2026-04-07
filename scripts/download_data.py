"""Download AAPL OHLCV data and compute features."""

import pandas as pd
import yfinance as yf
import yaml
from pathlib import Path

from src.utils.features import (
    build_feature_dataframe,
    rolling_zscore_normalize,
    FEATURE_NAMES,
)
from src.utils.seed import set_global_seed


def main():
    set_global_seed(42)

    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ticker = config["data"]["ticker"]
    # Download from 2007-01-01 to provide ~500 trading days of warm-up
    # before 2009-01-01 (warmup split start), which the 252-day rolling
    # z-score normalization needs for stable statistics.  The extra margin
    # ensures all 14 technical indicators (longest: MACD slow=26, BB=20,
    # RSI=14, realized_vol=20) plus the 252-day z-score window are fully
    # populated well before the warmup period begins.
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start="2007-01-01", end="2024-12-31", auto_adjust=True)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.31 returns MultiIndex for single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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
    # Warmup: 2009 (Guardian/Analyst warm-up, not used for RL training)
    # Train:  2010-2023 (~3,520 trading days — expanded from 756)
    # Val:    Jan-Jun 2024 (early stopping target)
    # Test:   Jul-Dec 2024 (held-out evaluation)
    splits = {
        "warmup": {"start": "2009-01-01", "end": "2009-12-31"},
        "train": {"start": "2010-01-01", "end": "2023-12-31"},
        "val": {"start": "2024-01-01", "end": "2024-06-30"},
        "test": {"start": "2024-07-01", "end": "2024-12-31"},
    }

    splits_path = project_root / "configs" / "data_splits.yaml"
    with open(splits_path, "w") as f:
        yaml.dump(splits, f, default_flow_style=False)
    print(f"Splits config saved to {splits_path}")

    # Print split sizes
    for split_name, dates in splits.items():
        mask = (features_norm.index >= dates["start"]) & (
            features_norm.index <= dates["end"]
        )
        n = mask.sum()
        print(f"  {split_name}: {n} trading days")


if __name__ == "__main__":
    main()
