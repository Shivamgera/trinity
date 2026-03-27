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
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
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
    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
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
        DataFrame with 14 feature columns, same index as input.
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
