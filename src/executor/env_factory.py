"""Factory functions for creating trading environments for PPO training.

This module provides helper utilities to construct single and vectorized
TradingEnv instances following the Stable-Baselines3 convention of
returning callables (environment factories) rather than already-built
environments.  This is required by DummyVecEnv/SubprocVecEnv.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

from src.executor.env import TradingEnv
from src.utils.data import load_numeric_features, load_raw_ohlcv


def make_trading_env(
    ticker: str = "AAPL",
    split: str = "train",
    lookback_window: int = 10,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    dsr_eta: float = 0.01,
    inaction_penalty: float = 0.0,
    inaction_threshold: int = 20,
    episode_length: int | None = None,
    random_start: bool = True,
    reward_type: str = "dsr",
) -> Callable[[], gym.Env]:
    """Create a factory callable for a single TradingEnv instance.

    Following the SB3 convention, this returns a *callable* that creates
    a new environment when invoked.  DummyVecEnv/SubprocVecEnv expect a
    list of such callables.

    Args:
        ticker: Ticker symbol. Must have processed data available.
        split: Data split to load: "train", "val", or "test".
        lookback_window: Days of history per observation.
        transaction_cost: Brokerage commission per unit position change.
        slippage: Bid-ask half-spread per unit position change.
        dsr_eta: Adaptation rate for Differential Sharpe Ratio.
        inaction_penalty: Per-step penalty for holding non-zero position
                          beyond inaction_threshold steps without trading.
        inaction_threshold: Steps before inaction_penalty activates.
        episode_length: Max steps per episode (None = full split).
        random_start: Randomize episode start position.
        reward_type: Reward function: "dsr" or "log_return".

    Returns:
        A callable ``() -> gym.Env`` that creates a fresh TradingEnv.
    """
    features_df = load_numeric_features(ticker=ticker, split=split)
    raw_ohlcv = load_raw_ohlcv(ticker=ticker)

    # Align close prices to feature index
    close_prices = raw_ohlcv["Close"].reindex(features_df.index).ffill()

    # Capture as numpy for thread-safety in SubprocVecEnv
    features_np = features_df.values.astype(np.float32)
    prices_np = close_prices.values.astype(np.float64)
    date_index = (
        features_df.index.strftime("%Y-%m-%d").tolist()
        if isinstance(features_df.index, pd.DatetimeIndex)
        else []
    )

    def _factory() -> gym.Env:
        env = TradingEnv(
            features=features_df,  # DataFrame so date_index is preserved
            prices=close_prices,
            lookback_window=lookback_window,
            transaction_cost=transaction_cost,
            slippage=slippage,
            dsr_eta=dsr_eta,
            inaction_penalty=inaction_penalty,
            inaction_threshold=inaction_threshold,
            episode_length=episode_length,
            random_start=random_start,
            reward_type=reward_type,
        )
        return env

    return _factory


def create_vec_env(
    n_envs: int = 8,
    ticker: str = "AAPL",
    split: str = "train",
    lookback_window: int = 10,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    dsr_eta: float = 0.01,
    inaction_penalty: float = 0.0,
    inaction_threshold: int = 20,
    episode_length: int | None = None,
    random_start: bool = True,
    reward_type: str = "dsr",
) -> list[Callable[[], gym.Env]]:
    """Create a list of env factory callables for use with DummyVecEnv.

    Example usage::

        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        env_fns = create_vec_env(n_envs=8)
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecNormalize(vec_env, clip_obs=10.0)

    Args:
        n_envs: Number of parallel environment instances.
        ticker: Ticker symbol.
        split: Data split to use.
        lookback_window: Days of history per observation.
        transaction_cost: Cost per unit position change.
        dsr_eta: Adaptation rate for DSR.
        inaction_penalty: Per-step penalty for non-zero position stasis.
        inaction_threshold: Steps before inaction_penalty activates.
        episode_length: Max steps per episode.
        random_start: Randomize episode start.
        reward_type: Reward function: "dsr" or "log_return".

    Returns:
        List of callables, each returning a fresh TradingEnv.
    """
    return [
        make_trading_env(
            ticker=ticker,
            split=split,
            lookback_window=lookback_window,
            transaction_cost=transaction_cost,
            slippage=slippage,
            dsr_eta=dsr_eta,
            inaction_penalty=inaction_penalty,
            inaction_threshold=inaction_threshold,
            episode_length=episode_length,
            random_start=random_start,
            reward_type=reward_type,
        )
        for _ in range(n_envs)
    ]
