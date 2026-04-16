"""Custom Gymnasium trading environment for the Executor agent.

Architecture:
    The Executor is a PPO agent that observes z-normalized market features
    and outputs a position target: flat (0), long (1), or short (2).
    The environment handles position management, transaction costs, and
    reward computation via the Differential Sharpe Ratio.

Observation Space:
    Box of shape (lookback_window * n_features + portfolio_state_dim,)
    - lookback_window (default 10): days of historical features
    - n_features (default 14): z-normalized technical indicators
    - portfolio_state (3): [current_position, unrealized_pnl, time_since_last_trade]
    Total default: 10 * 14 + 3 = 143

Action Space:
    Discrete(3): {0=flat, 1=long, 2=short}
    These are POSITION TARGETS, not trades. The environment computes
    the trade needed: trade = target_position - current_position.

Reward:
    Differential Sharpe Ratio (Moody & Saffell, 2001) with transaction
    cost penalty.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from src.executor.rewards import (
    CVaRPenalizedReward,
    DifferentialSharpeReward,
    LogReturnReward,
    MeanVarianceReward,
    SortinoReward,
)

# Position encoding: maps action index to position value
ACTION_TO_POSITION = {0: 0.0, 1: 1.0, 2: -1.0}  # flat, long, short


class TradingEnv(gym.Env):
    """Gymnasium trading environment for PPO-based Executor agent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame | np.ndarray,
        prices: pd.Series | np.ndarray,
        lookback_window: int = 10,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        dsr_eta: float = 0.01,
        inaction_penalty: float = 0.0,
        inaction_threshold: int = 20,
        episode_length: int | None = None,
        random_start: bool = True,
        reward_type: str = "dsr",
    ):
        """
        Args:
            features: Z-normalized feature matrix, shape (T, n_features).
                      If DataFrame, will be converted to numpy.
                      If DataFrame with DatetimeIndex, date strings are
                      preserved for C-Gate date synchronization.
            prices: Close prices for return computation, shape (T,).
                    Must align with features rows.
            lookback_window: Number of past days to include in observation.
            transaction_cost: Brokerage commission per unit of position change
                              (e.g., 0.001 = 10bps). Paid on every trade.
            slippage: Bid-ask half-spread per unit of position change
                      (e.g., 0.0005 = 5bps). Models the cost of crossing the
                      book. Applied on top of transaction_cost, so total round-
                      trip friction is 2 * (transaction_cost + slippage).
                      Default 5bps is conservative for AAPL (typical spread
                      is 1-3bps), providing a safety margin against the
                      "infinite liquidity" simulation artefact.
            dsr_eta: Adaptation rate for Differential Sharpe Ratio.
            inaction_penalty: Per-step penalty applied when the agent holds a
                              non-zero position (long or short) without trading
                              for more than `inaction_threshold` consecutive
                              steps. Discourages always-short / always-long
                              mode collapse without penalizing legitimate
                              trend-following. Default 0.0 (disabled).
            inaction_threshold: Number of steps of unchanged non-zero position
                                before the inaction_penalty kicks in.
                                Default 20 (≈ 1 trading month).
            episode_length: Max steps per episode. None = use all remaining data.
            random_start: If True, randomize starting index on reset.
            reward_type: Reward function to use. "dsr" for Differential
                         Sharpe Ratio (Moody & Saffell 2001), "log_return"
                         for simple log-return reward.  Default "dsr" for
                         backward compatibility.
        """
        super().__init__()

        # Preserve date index for C-Gate synchronization (Phase 4).
        # When the C-Gate integration runs the Executor through TradingEnv,
        # it needs to know which calendar date corresponds to each step
        # so it can look up the Analyst's precomputed signal for that date.
        if isinstance(features, pd.DataFrame) and isinstance(features.index, pd.DatetimeIndex):
            self._date_index: list[str] = features.index.strftime("%Y-%m-%d").tolist()
        else:
            self._date_index = []

        # Convert to numpy
        if isinstance(features, pd.DataFrame):
            self._feature_values = features.values.astype(np.float32)
        else:
            self._feature_values = np.asarray(features, dtype=np.float32)

        if isinstance(prices, pd.Series):
            self._prices = prices.values.astype(np.float64)
        else:
            self._prices = np.asarray(prices, dtype=np.float64)

        assert len(self._feature_values) == len(self._prices), (
            f"Features ({len(self._feature_values)}) and prices ({len(self._prices)}) must align"
        )

        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.inaction_penalty = inaction_penalty
        self.inaction_threshold = inaction_threshold
        self.episode_length = episode_length
        self.random_start = random_start

        self.n_features = self._feature_values.shape[1]
        self.portfolio_state_dim = 3  # [position, unrealized_pnl, time_since_trade]

        # Observation: flattened lookback window + portfolio state
        self.obs_dim = self.lookback_window * self.n_features + self.portfolio_state_dim

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)  # flat=0, long=1, short=2

        # Reward function
        self.reward_type = reward_type
        if reward_type == "log_return":
            self._reward_fn = LogReturnReward()
        elif reward_type == "dsr":
            self._reward_fn = DifferentialSharpeReward(eta=dsr_eta)
        elif reward_type == "sortino":
            self._reward_fn = SortinoReward()
        elif reward_type == "mean_variance":
            self._reward_fn = MeanVarianceReward()
        elif reward_type == "cvar":
            self._reward_fn = CVaRPenalizedReward()
        else:
            raise ValueError(
                f"Unknown reward_type '{reward_type}'. "
                f"Expected 'dsr', 'log_return', 'sortino', 'mean_variance', or 'cvar'."
            )

        # State variables (initialized in reset)
        self._current_step: int = 0
        self._start_step: int = 0
        self._position: float = 0.0  # -1, 0, or 1
        self._entry_price: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._time_since_trade: int = 0
        self._total_pnl: float = 0.0

    @property
    def current_date(self) -> str | None:
        """Return the calendar date (YYYY-MM-DD) of the current step.

        Returns None if the environment was constructed from raw numpy
        arrays without a DatetimeIndex.  The C-Gate integration loop
        (Phase 4) uses this to look up the Analyst's precomputed signal
        for the corresponding trading day.
        """
        if self._date_index and 0 <= self._current_step < len(self._date_index):
            return self._date_index[self._current_step]
        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Optional seed for reproducibility.
            options: Optional dict; may contain 'start_index' to override random start.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Determine starting index
        min_start = self.lookback_window
        max_start = len(self._feature_values) - 1

        if self.episode_length is not None:
            max_start = len(self._feature_values) - self.episode_length

        max_start = max(max_start, min_start + 1)

        if options and "start_index" in options:
            self._start_step = max(options["start_index"], min_start)
        elif self.random_start:
            self._start_step = self.np_random.integers(min_start, max_start)
        else:
            self._start_step = min_start

        self._current_step = self._start_step

        # Reset portfolio state
        self._position = 0.0
        self._entry_price = 0.0
        self._unrealized_pnl = 0.0
        self._time_since_trade = 0
        self._total_pnl = 0.0

        # Reset reward function
        self._reward_fn.reset()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one trading step.

        Args:
            action: Position target: 0=flat, 1=long, 2=short.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Map action to target position
        target_position = ACTION_TO_POSITION[int(action)]

        # Compute trade
        trade = target_position - self._position

        # Transaction cost + slippage (bid-ask half-spread)
        tc = (self.transaction_cost + self.slippage) * abs(trade)

        # Update position BEFORE computing return: the agent's action at
        # step t selects a position, and the return from t to t+1 accrues
        # to that chosen position.  Previously the return was attributed
        # to the *old* position, creating a one-step delay where TC was
        # paid immediately but P&L only arrived next step — a systematic
        # negative bias.
        if trade != 0:
            self._position = target_position
            self._entry_price = self._prices[self._current_step]
            self._time_since_trade = 0
        else:
            self._time_since_trade += 1

        # Current price (before advancing)
        current_price = self._prices[self._current_step]

        # Advance to next step
        self._current_step += 1

        # Check termination
        terminated = self._current_step >= len(self._feature_values) - 1
        truncated = False
        if self.episode_length is not None:
            steps_taken = self._current_step - self._start_step
            if steps_taken >= self.episode_length:
                truncated = True

        next_price = self._prices[self._current_step]

        # Compute portfolio return using the CURRENT (post-trade) position
        price_return = (next_price - current_price) / current_price
        portfolio_return = self._position * price_return - tc

        # Update PnL tracking
        self._total_pnl += portfolio_return

        # Update unrealized PnL
        if self._position != 0 and self._entry_price > 0:
            self._unrealized_pnl = (
                self._position * (next_price - self._entry_price) / self._entry_price
            )
        else:
            self._unrealized_pnl = 0.0

        # Inaction penalty: discourages constant-position mode collapse.
        # Applied when agent holds a non-zero position beyond the threshold
        # without changing it, so legitimate trend-following is not penalized
        # during the initial hold window.
        if (
            self.inaction_penalty > 0.0
            and self._position != 0.0
            and self._time_since_trade > self.inaction_threshold
        ):
            portfolio_return -= self.inaction_penalty

        # Compute reward via Differential Sharpe Ratio
        reward = self._reward_fn.compute(portfolio_return)

        obs = self._get_observation()
        info = self._get_info()
        info["portfolio_return"] = portfolio_return
        info["trade"] = trade
        info["transaction_cost"] = tc

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector.

        Returns:
            Flat numpy array of shape (obs_dim,).
        """
        # Feature lookback window
        start = self._current_step - self.lookback_window
        end = self._current_step
        feature_window = self._feature_values[start:end].flatten()

        # Portfolio state
        portfolio_state = np.array(
            [
                self._position,
                self._unrealized_pnl,
                min(self._time_since_trade / 20.0, 1.0),  # Normalized to [0, 1]
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([feature_window, portfolio_state])
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Return info dictionary.

        Includes ``"date"`` when the environment was built from a
        DataFrame with a DatetimeIndex, enabling the C-Gate integration
        (Phase 4) to synchronize Analyst signals with Executor steps.
        """
        info: dict[str, Any] = {
            "step": self._current_step,
            "position": self._position,
            "unrealized_pnl": self._unrealized_pnl,
            "total_pnl": self._total_pnl,
            "time_since_trade": self._time_since_trade,
            "price": float(self._prices[self._current_step]),
        }
        if self.current_date is not None:
            info["date"] = self.current_date
        return info
