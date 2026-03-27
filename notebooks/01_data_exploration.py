"""Data Exploration — Robust Trinity.

Verify numeric features, headlines, and channel independence.
Generates publication-ready plots for inspection.

Usage:
    python3 notebooks/01_data_exploration.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import (
    get_feature_names,
    load_headlines,
    load_numeric_features,
    load_raw_ohlcv,
    verify_channel_independence,
)

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def explore_numeric_features():
    """Load and summarize numeric features."""
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

    # Descriptive statistics
    print("\nTrain set descriptive statistics:")
    print(features_train.describe().round(3).to_string())

    return features_train, features_test, raw


def plot_price_and_volume(raw: pd.DataFrame):
    """Plot raw AAPL price and volume with train/test split markers."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(raw.index, raw["Close"], linewidth=0.8)
    axes[0].axvline(pd.Timestamp("2021-01-01"), color="green", linestyle="--", label="Train start")
    axes[0].axvline(pd.Timestamp("2024-01-01"), color="orange", linestyle="--", label="Val start")
    axes[0].axvline(pd.Timestamp("2024-07-01"), color="red", linestyle="--", label="Test start")
    axes[0].set_title("AAPL Close Price")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()

    axes[1].bar(raw.index, raw["Volume"], width=1, alpha=0.5)
    axes[1].set_title("AAPL Volume")
    axes[1].set_ylabel("Volume")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "price_and_volume.png", dpi=150)
    plt.close()
    print("Saved price_and_volume.png")


def plot_feature_distributions(features_train: pd.DataFrame):
    """Plot histograms of all z-normalized features."""
    feature_names = get_feature_names()
    nrows = 4
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))

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
    plt.savefig(OUT_DIR / "feature_distributions.png", dpi=150)
    plt.close()
    print("Saved feature_distributions.png")


def plot_feature_correlations(features_train: pd.DataFrame):
    """Plot feature correlation heatmap."""
    corr = features_train.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        xticklabels=True, yticklabels=True, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Train Set)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_correlations.png", dpi=150)
    plt.close()
    print("Saved feature_correlations.png")

    # Flag highly correlated pairs
    print("\nHighly correlated pairs (|r| > 0.8):")
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                print(f"  {corr.columns[i]} <-> {corr.columns[j]}: {r:.3f}")


def explore_headlines():
    """Load and summarize headlines."""
    print("\n" + "=" * 60)
    print("HEADLINES")
    print("=" * 60)
    headlines = load_headlines("AAPL")
    print(f"Total headlines: {len(headlines)}")
    print(f"Date range: {headlines[0]['date']} to {headlines[-1]['date']}")

    # Source distribution
    sources = {}
    for h in headlines:
        sources[h["source"]] = sources.get(h["source"], 0) + 1
    print("\nHeadlines by source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # Sample headlines
    print("\nSample headlines (first 10):")
    for h in headlines[:10]:
        print(f"  [{h['date']}] ({h['source']}) {h['headline'][:100]}")

    # Headline length statistics
    lengths = [len(h["headline"]) for h in headlines]
    print(f"\nHeadline length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")


def explore_phrasebank():
    """Load and summarize FinancialPhraseBank."""
    import csv

    phrasebank_path = Path(__file__).parent.parent / "data" / "raw" / "financial_phrasebank.csv"
    if not phrasebank_path.exists():
        print("\nFinancialPhraseBank not found — skipping.")
        return

    print("\n" + "=" * 60)
    print("FINANCIAL PHRASEBANK")
    print("=" * 60)

    with open(phrasebank_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total sentences: {len(rows)}")
    labels = [r["label"] for r in rows]
    from collections import Counter
    counts = Counter(labels)
    print(f"Label distribution: {dict(counts)}")

    # Samples per label
    for label in ["positive", "negative", "neutral"]:
        samples = [r for r in rows if r["label"] == label][:3]
        print(f"\n  {label.upper()} samples:")
        for s in samples:
            print(f"    {s['sentence'][:100]}")


def main():
    features_train, features_test, raw = explore_numeric_features()
    plot_price_and_volume(raw)
    plot_feature_distributions(features_train)
    plot_feature_correlations(features_train)
    explore_headlines()
    explore_phrasebank()
    print("\n" + "=" * 60)
    print("DATA EXPLORATION COMPLETE")
    print(f"Check {OUT_DIR}/ for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
