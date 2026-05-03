"""Download OHLCV data and compute features for one or more tickers.

Usage:
    python3 -m scripts.download_data                  # AAPL only (default)
    python3 -m scripts.download_data --multi-ticker   # AAPL + training augmentation tickers
"""

import argparse

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

# Primary ticker (deployment target)
PRIMARY_TICKER = "AAPL"

# Additional tickers for multi-ticker training augmentation.
# Selected for regime diversity: SPY (broad market), MSFT (similar sector,
# different drawdown profile), GOOGL (tech, more volatile), AMZN (went
# sideways/down 2021-2023, teaches defensive positioning).
AUGMENTATION_TICKERS = ["MSFT", "GOOGL", "SPY", "AMZN"]


def download_ticker(ticker: str, project_root: Path) -> None:
    """Download OHLCV data for a single ticker and compute z-normalized features."""
    print(f"\n{'=' * 60}")
    print(f"Downloading {ticker} data...")
    print(f"{'=' * 60}")

    df = yf.download(ticker, start="2007-01-01", end="2024-12-31", auto_adjust=True)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"Downloaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Build features
    print("Computing features...")
    features = build_feature_dataframe(df)

    # Apply rolling z-score normalization (per-ticker, not cross-ticker)
    print("Applying rolling z-score normalization...")
    features_norm = rolling_zscore_normalize(features, window=252)

    # Drop NaN rows (from rolling windows)
    features_norm = features_norm.dropna()

    print(f"Final feature matrix: {features_norm.shape}")
    print(f"Date range: {features_norm.index[0]} to {features_norm.index[-1]}")
    print(f"Features: {list(features_norm.columns)}")

    # Save processed features
    output_path = project_root / "data" / "processed" / f"{ticker.lower()}_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_norm.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # Save raw OHLCV
    raw_path = project_root / "data" / "raw" / f"{ticker.lower()}_ohlcv.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path)
    print(f"Raw OHLCV saved to {raw_path}")

    return features_norm


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data and compute features")
    parser.add_argument(
        "--multi-ticker",
        action="store_true",
        help="Download augmentation tickers (MSFT, GOOGL, SPY, AMZN) in addition to AAPL",
    )
    args = parser.parse_args()

    set_global_seed(42)

    project_root = Path(__file__).parent.parent

    # Determine tickers to download
    tickers = [PRIMARY_TICKER]
    if args.multi_ticker:
        tickers += AUGMENTATION_TICKERS
        print(f"Multi-ticker mode: downloading {tickers}")

    # Download each ticker
    for ticker in tickers:
        features_norm = download_ticker(ticker, project_root)

    # Create/update split config (same dates for all tickers)
    splits = {
        "warmup": {"start": "2009-01-01", "end": "2009-12-31"},
        "train": {"start": "2010-01-01", "end": "2023-12-31"},
        "val": {"start": "2024-01-01", "end": "2024-06-30"},
        "test": {"start": "2024-07-01", "end": "2024-12-31"},
    }

    splits_path = project_root / "configs" / "data_splits.yaml"
    with open(splits_path, "w") as f:
        yaml.dump(splits, f, default_flow_style=False)
    print(f"\nSplits config saved to {splits_path}")

    # Print split sizes for the primary ticker
    primary_path = (
        project_root / "data" / "processed" / f"{PRIMARY_TICKER.lower()}_features.parquet"
    )
    primary_features = pd.read_parquet(primary_path)
    for split_name, dates in splits.items():
        mask = (primary_features.index >= dates["start"]) & (primary_features.index <= dates["end"])
        n = mask.sum()
        print(f"  {split_name}: {n} trading days")


if __name__ == "__main__":
    main()
