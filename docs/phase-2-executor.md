# Phase 2: Executor Agent — PPO Training

**Project:** Robust Trinity — Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Timeline:** Weeks 3-5, ~25-35 hours total
**Goal:** Build the PPO-based Executor agent — the reinforcement learning component of the Robust Trinity. This includes a custom Gymnasium trading environment, PPO training pipeline, policy distribution extraction (critical for the C-Gate), hyperparameter tuning, and out-of-sample evaluation.

**Prerequisites:** Phase 0 (project setup) and Phase 1 (data pipeline) must be complete. Specifically:
- `data/processed/aapl_features.parquet` exists with ~14 z-normalized features
- `configs/data_splits.yaml` defines train/test date ranges
- `src/utils/data.py` has working `load_numeric_features()` function
- `src/utils/seed.py` has working `set_global_seed()` function
- SB3, Gymnasium, PyTorch, W&B are all verified working

---

## P2-T1: Build Custom Gymnasium Trading Environment

**Estimated time:** ~3 hours
**Dependencies:** P1-T1 (numeric features must be available in `data/processed/aapl_features.parquet`)

### Context

The Robust Trinity system has an **Executor agent** that is a PPO-based reinforcement learning policy. It operates on a **position-target action space**:
- Action 0 = **flat** (no position, all cash)
- Action 1 = **long** (fully long the asset)
- Action 2 = **short** (fully short the asset)

The agent does NOT output trade sizes — it outputs a *target position*, and the environment computes the trade needed to reach that position from the current one.

The observation consists of:
- A **lookback window** of z-normalized numeric features (30 days × 14 features = 420 dims)
- **Portfolio state**: [current_position, unrealized_pnl, time_since_last_trade] (3 dims)
- Total observation dimension: 423

The reward function uses the **Differential Sharpe Ratio** (Moody & Saffell, 2001), which provides an online, incremental estimate of the Sharpe ratio at each step. This is superior to raw PnL as a reward because it penalizes volatility.

Key design decisions:
- Transaction cost: 10 basis points (0.001) per unit of position change
- Episodes span the full training data or a configurable rolling window
- Reset randomizes the starting point within the training data (with sufficient lookback buffer)
- The environment outputs the policy distribution over {flat, long, short} — this is critical for the C-Gate which computes Δ = 1 - π_RL(d_LLM) to measure how much probability the Executor assigns to the Analyst's chosen action

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Create a fully functional Gymnasium-compatible trading environment with Differential Sharpe Ratio reward, position-target action space, and proper observation construction.

### Detailed Instructions

**Step 1: Create the Differential Sharpe Ratio reward class**

Create `src/executor/rewards.py`:

```python
"""Reward functions for the trading environment.

The Differential Sharpe Ratio (DSR) provides an online, incremental
estimate of the Sharpe ratio at each timestep. It was introduced by
Moody & Saffell (2001) and is well-suited for RL because it gives
meaningful per-step rewards rather than only episode-level metrics.

DSR formula:
    D_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

Where:
    A_t = A_{t-1} + η * (R_t - A_{t-1})          [EMA of returns]
    B_t = B_{t-1} + η * (R_t^2 - B_{t-1})        [EMA of squared returns]
    η = adaptation rate (e.g., 0.01)
    R_t = portfolio return at time t
"""

import numpy as np


class DifferentialSharpeReward:
    """Computes the Differential Sharpe Ratio at each timestep.

    This provides a per-step reward that approximates the marginal
    contribution of the current return to the overall Sharpe ratio.
    """

    def __init__(self, eta: float = 0.01):
        """
        Args:
            eta: Adaptation rate for EMA estimates. Smaller values
                 give smoother but slower-adapting estimates.
                 Typical range: 0.001 to 0.05.
        """
        self.eta = eta
        self.reset()

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self._step = 0

    def compute(self, portfolio_return: float) -> float:
        """Compute the Differential Sharpe Ratio for this timestep.

        Args:
            portfolio_return: The portfolio return R_t at this step.

        Returns:
            The DSR reward value.
        """
        self._step += 1

        # For the first few steps, use simple reward to bootstrap
        if self._step < 3:
            self.A = self.A + self.eta * (portfolio_return - self.A)
            self.B = self.B + self.eta * (portfolio_return**2 - self.B)
            return portfolio_return

        # Compute deltas
        delta_A = portfolio_return - self.A
        delta_B = portfolio_return**2 - self.B

        # Denominator: (B - A^2)^{3/2}
        variance_est = self.B - self.A**2
        if variance_est <= 1e-12:
            # Avoid division by zero; fall back to simple return
            dsr = portfolio_return
        else:
            denominator = variance_est ** 1.5
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / denominator

        # Update EMAs
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B

        return float(dsr)
```

**Step 2: Create the trading environment**

Create `src/executor/env.py`:

```python
"""Custom Gymnasium trading environment for the Executor agent.

Architecture:
    The Executor is a PPO agent that observes z-normalized market features
    and outputs a position target: flat (0), long (1), or short (2).
    The environment handles position management, transaction costs, and
    reward computation via the Differential Sharpe Ratio.

Observation Space:
    Box of shape (lookback_window * n_features + portfolio_state_dim,)
    - lookback_window (default 30): days of historical features
    - n_features (default 14): z-normalized technical indicators
    - portfolio_state (3): [current_position, unrealized_pnl, time_since_last_trade]
    Total default: 30 * 14 + 3 = 423

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

from src.executor.rewards import DifferentialSharpeReward


# Position encoding: maps action index to position value
ACTION_TO_POSITION = {0: 0.0, 1: 1.0, 2: -1.0}  # flat, long, short


class TradingEnv(gym.Env):
    """Gymnasium trading environment for PPO-based Executor agent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame | np.ndarray,
        prices: pd.Series | np.ndarray,
        lookback_window: int = 30,
        transaction_cost: float = 0.001,
        dsr_eta: float = 0.01,
        episode_length: int | None = None,
        random_start: bool = True,
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
            transaction_cost: Cost per unit of position change (e.g., 0.001 = 10bps).
            dsr_eta: Adaptation rate for Differential Sharpe Ratio.
            episode_length: Max steps per episode. None = use all remaining data.
            random_start: If True, randomize starting index on reset.
        """
        super().__init__()

        # Preserve date index for C-Gate synchronization (Phase 4).
        # When the C-Gate integration runs the Executor through TradingEnv,
        # it needs to know which calendar date corresponds to each step
        # so it can look up the Analyst's precomputed signal for that date.
        if isinstance(features, pd.DataFrame) and isinstance(
            features.index, pd.DatetimeIndex
        ):
            self._date_index: list[str] = (
                features.index.strftime("%Y-%m-%d").tolist()
            )
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
        self._reward_fn = DifferentialSharpeReward(eta=dsr_eta)

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

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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

        # Transaction cost
        tc = self.transaction_cost * abs(trade)

        # Current and next price
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

        # Compute portfolio return
        price_return = (next_price - current_price) / current_price
        portfolio_return = self._position * price_return - tc

        # Update PnL tracking
        self._total_pnl += portfolio_return

        # Update position
        if trade != 0:
            self._position = target_position
            self._entry_price = current_price
            self._time_since_trade = 0
        else:
            self._time_since_trade += 1

        # Update unrealized PnL
        if self._position != 0 and self._entry_price > 0:
            self._unrealized_pnl = self._position * (next_price - self._entry_price) / self._entry_price
        else:
            self._unrealized_pnl = 0.0

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
        portfolio_state = np.array([
            self._position,
            self._unrealized_pnl,
            min(self._time_since_trade / 20.0, 1.0),  # Normalized to [0, 1]
        ], dtype=np.float32)

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
```

**Step 3: Create tests `tests/test_env.py`**

```python
"""Tests for the custom trading environment."""

import numpy as np
import pytest
import gymnasium as gym

from src.executor.env import TradingEnv, ACTION_TO_POSITION
from src.executor.rewards import DifferentialSharpeReward
from src.utils.seed import set_global_seed


class TestDifferentialSharpeReward:
    def test_positive_returns(self):
        dsr = DifferentialSharpeReward(eta=0.01)
        # Consistently positive returns should give positive rewards
        rewards = [dsr.compute(0.01) for _ in range(100)]
        assert sum(r > 0 for r in rewards[10:]) > len(rewards[10:]) * 0.5

    def test_reset(self):
        dsr = DifferentialSharpeReward(eta=0.01)
        dsr.compute(0.05)
        dsr.compute(0.03)
        dsr.reset()
        assert dsr.A == 0.0
        assert dsr.B == 0.0


class TestTradingEnv:
    @pytest.fixture
    def env(self):
        """Create a simple test environment."""
        set_global_seed(42)
        T = 200
        n_features = 14
        features = np.random.randn(T, n_features).astype(np.float32)
        prices = 100 + np.cumsum(np.random.randn(T) * 0.5)
        prices = np.maximum(prices, 10)  # Prevent negative prices
        return TradingEnv(
            features=features,
            prices=prices,
            lookback_window=30,
            random_start=False,
        )

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset(seed=42)
        expected_dim = 30 * 14 + 3  # lookback * features + portfolio state
        assert obs.shape == (expected_dim,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, env):
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)  # Go long
        assert obs.shape == (30 * 14 + 3,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self, env):
        assert env.action_space.n == 3
        for action in [0, 1, 2]:
            env.reset(seed=42)
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.isnan(reward)

    def test_observation_space(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_position_tracking(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(1)  # Go long
        assert info["position"] == 1.0
        _, _, _, _, info = env.step(2)  # Go short
        assert info["position"] == -1.0
        _, _, _, _, info = env.step(0)  # Go flat
        assert info["position"] == 0.0

    def test_transaction_cost(self, env):
        env.reset(seed=42)
        _, _, _, _, info1 = env.step(0)  # Stay flat (no trade)
        assert info1["trade"] == 0.0
        assert info1["transaction_cost"] == 0.0
        _, _, _, _, info2 = env.step(1)  # Flat→Long = trade of 1.0
        assert info2["trade"] == 1.0
        assert info2["transaction_cost"] == pytest.approx(0.001)

    def test_episode_terminates(self, env):
        env.reset(seed=42)
        terminated = False
        truncated = False
        steps = 0
        while not terminated and not truncated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if steps > 500:
                break
        assert terminated or truncated, "Episode should eventually end"

    def test_date_tracking(self):
        """When features is a DataFrame with DatetimeIndex, info['date'] is set."""
        set_global_seed(42)
        T = 200
        n_features = 14
        dates = pd.bdate_range("2023-01-01", periods=T)
        features = pd.DataFrame(
            np.random.randn(T, n_features).astype(np.float32),
            index=dates,
        )
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(T) * 0.5),
            index=dates,
        )
        prices = prices.clip(lower=10)
        env = TradingEnv(
            features=features,
            prices=prices,
            lookback_window=30,
            random_start=False,
        )
        _, info = env.reset(seed=42)
        assert "date" in info, "info dict must contain 'date' when built from DataFrame"
        assert isinstance(info["date"], str)
        # Stepping should advance the date
        _, _, _, _, info2 = env.step(1)
        assert info2["date"] > info["date"]

    def test_no_date_when_numpy(self, env):
        """When features is raw numpy, info should NOT contain 'date'."""
        _, info = env.reset(seed=42)
        assert "date" not in info

    def test_gymnasium_compatibility(self, env):
        """Verify the env passes gymnasium's check_env."""
        from gymnasium.utils.env_checker import check_env
        # check_env may print warnings; that's fine
        try:
            check_env(env.unwrapped, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium check_env failed: {e}")
```

**Step 4: Run tests**

```bash
pytest tests/test_env.py -v
```

**Step 5: Commit**

```bash
git add src/executor/rewards.py src/executor/env.py tests/test_env.py
git commit -m "P2-T1: custom Gymnasium trading environment with Differential Sharpe Ratio reward"
```

### Acceptance Criteria

1. `TradingEnv` is a valid `gymnasium.Env` subclass that passes `check_env`
2. Observation shape is `(lookback_window * n_features + 3,)` = `(423,)` with default params
3. Action space is `Discrete(3)` mapping to {flat, long, short}
4. `step()` returns `(obs, reward, terminated, truncated, info)` with correct types
5. Transaction costs are applied correctly (0.1% per unit of position change)
6. Differential Sharpe Ratio reward is non-NaN for all steps
7. Episodes terminate when data runs out
8. Position tracking is correct through action sequences
9. When constructed from a DataFrame with DatetimeIndex, `_get_info()` includes `"date"` as a `"YYYY-MM-DD"` string corresponding to the current step. When constructed from raw numpy arrays, `"date"` is absent from `info`.
10. `current_date` property returns the correct calendar date for the current step
11. All tests in `tests/test_env.py` pass

### Files to Create

- `src/executor/rewards.py`
- `src/executor/env.py`
- `tests/test_env.py`

### Files to Modify

- None

### Human Checkpoint

- Review the reward function implementation against the Moody & Saffell (2001) paper
- Verify the observation construction makes sense (lookback window + portfolio state)
- Check that the action-to-position mapping is correct (0=flat, 1=long, 2=short)
- Run `check_env` and confirm no errors

---

## P2-T2: Implement and Verify PPO Training Pipeline

**Estimated time:** ~3 hours
**Dependencies:** P2-T1 (trading environment must exist), P1-T1 (numeric features must be available)

### Context

The Robust Trinity's **Executor agent** uses PPO (Proximal Policy Optimization) from Stable-Baselines3 (SB3). The Executor is trained offline on historical AAPL data and then frozen — there is no online learning. The frozen policy's output distribution over {flat, long, short} will later be used to compute the C-Gate divergence Δ = 1 - π_RL(d_LLM), where d_LLM is the Analyst's chosen action.

The custom trading environment (`src/executor/env.py`, built in P2-T1) provides:
- Observation: 30-day lookback window of 14 z-normalized features + 3 portfolio state dims = 423 dimensions
- Action space: Discrete(3) — position targets {flat=0, long=1, short=2}
- Reward: Differential Sharpe Ratio with transaction cost penalty

PPO hyperparameters for the initial run:
- Policy network: MLP with two hidden layers of 64 units each (separate for policy and value)
- Learning rate: 3e-4
- n_steps: 2048 (steps per environment before update)
- batch_size: 64
- n_epochs: 10 (PPO epochs per update)
- gamma: 0.99 (discount factor)
- gae_lambda: 0.95 (GAE parameter)
- clip_range: 0.2 (PPO clipping)
- ent_coef: 0.01 (entropy bonus for exploration)
- 8 parallel environments
- 500K total timesteps for initial run

SB3's `VecNormalize` wrapper should be used to normalize observations (even though our features are already z-normalized, the portfolio state dims and the concatenation benefit from additional normalization).

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Create the PPO training script, train an initial model, log metrics to W&B, and save the model and normalization statistics.

### Detailed Instructions

**Step 1: Create environment factory `src/executor/env_factory.py`**

SB3 requires a function that creates environment instances. This factory loads the data and creates `TradingEnv` instances.

```python
"""Environment factory for SB3 vectorized training."""

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
    lookback_window: int = 30,
    transaction_cost: float = 0.001,
    dsr_eta: float = 0.01,
    episode_length: int | None = None,
    random_start: bool = True,
    seed: int = 42,
    env_index: int = 0,
) -> Callable[[], gym.Env]:
    """Return a callable that creates a TradingEnv instance.

    This is the pattern required by SB3's make_vec_env and SubprocVecEnv.

    Args:
        ticker: Stock ticker.
        split: Data split to use.
        lookback_window: Days of history in observation.
        transaction_cost: Cost per unit position change.
        dsr_eta: DSR adaptation rate.
        episode_length: Max steps per episode (None = full data).
        random_start: Randomize episode start.
        seed: Base random seed.
        env_index: Index for this env in vectorized setup.

    Returns:
        A callable that creates a TradingEnv.
    """
    def _make() -> gym.Env:
        features = load_numeric_features(ticker, split=split)
        raw = load_raw_ohlcv(ticker)

        # Align prices with features
        prices = raw.loc[features.index, "Close"]

        env = TradingEnv(
            features=features,
            prices=prices,
            lookback_window=lookback_window,
            transaction_cost=transaction_cost,
            dsr_eta=dsr_eta,
            episode_length=episode_length,
            random_start=random_start,
        )
        env.reset(seed=seed + env_index)
        return env

    return _make


def create_vec_env(
    n_envs: int = 8,
    ticker: str = "AAPL",
    split: str = "train",
    lookback_window: int = 30,
    seed: int = 42,
    **kwargs,
) -> list[Callable[[], gym.Env]]:
    """Create a list of env factory callables for SB3 SubprocVecEnv.

    Args:
        n_envs: Number of parallel environments.
        ticker: Stock ticker.
        split: Data split.
        lookback_window: Observation lookback.
        seed: Base seed.
        **kwargs: Additional args passed to make_trading_env.

    Returns:
        List of callables, one per environment.
    """
    return [
        make_trading_env(
            ticker=ticker,
            split=split,
            lookback_window=lookback_window,
            seed=seed,
            env_index=i,
            **kwargs,
        )
        for i in range(n_envs)
    ]
```

**Step 2: Create the training script `src/executor/train.py`**

```python
"""PPO training pipeline for the Executor agent.

Usage:
    python -m src.executor.train

This trains a PPO agent on the AAPL trading environment and saves
the model and VecNormalize statistics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb

from src.executor.env_factory import create_vec_env
from src.utils.seed import set_global_seed
from src.utils.logging import init_wandb, log_metrics, finish_wandb


class WandbCallback(BaseCallback):
    """Custom callback for logging training metrics to W&B."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            # Training metrics from the logger
            metrics = {}
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                metrics["train/mean_episode_reward"] = np.mean(ep_rewards)
                metrics["train/mean_episode_length"] = np.mean(ep_lengths)

            # Log available training stats
            if hasattr(self.model, "logger") and self.model.logger is not None:
                for key in ["train/policy_loss", "train/value_loss",
                           "train/entropy_loss", "train/approx_kl",
                           "train/clip_fraction"]:
                    # These are logged by SB3 internally
                    pass

            metrics["train/timesteps"] = self.num_timesteps

            if metrics:
                log_metrics(metrics, step=self.num_timesteps)

        return True


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    lookback_window: int = 30,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    seed: int = 42,
    run_name: str | None = None,
) -> Path:
    """Train a PPO Executor agent.

    Args:
        total_timesteps: Total training steps across all environments.
        n_envs: Number of parallel environments.
        lookback_window: Days of feature history in observation.
        learning_rate: PPO learning rate.
        n_steps: Steps per env between PPO updates.
        batch_size: Minibatch size for PPO updates.
        n_epochs: Number of PPO optimization epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clipping parameter.
        ent_coef: Entropy coefficient for exploration.
        seed: Random seed.
        run_name: Optional W&B run name override.

    Returns:
        Path to the saved model directory.
    """
    set_global_seed(seed)

    # Config dict for logging
    config = {
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "lookback_window": lookback_window,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "seed": seed,
    }

    # Initialize W&B
    run = init_wandb(
        phase="P2",
        component="executor",
        config=config,
    )

    # Create vectorized environment
    env_fns = create_vec_env(
        n_envs=n_envs,
        split="train",
        lookback_window=lookback_window,
        seed=seed,
    )
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        seed=seed,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
    )

    # Train
    print(f"Starting PPO training: {total_timesteps} timesteps, {n_envs} envs")
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(),
        progress_bar=True,
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("experiments") / "executor" / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.zip"
    model.save(str(model_path))

    # Save VecNormalize stats
    vec_norm_path = save_dir / "vec_normalize.pkl"
    vec_env.save(str(vec_norm_path))

    print(f"Model saved to {model_path}")
    print(f"VecNormalize saved to {vec_norm_path}")

    # Log final info
    log_metrics({
        "train/total_timesteps": total_timesteps,
        "train/model_path": str(model_path),
    })

    finish_wandb()
    vec_env.close()

    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO Executor agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        lookback_window=args.lookback,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

**Step 3: Run training**

```bash
python -m src.executor.train --timesteps 500000
```

This will take some time depending on hardware (likely 10-30 minutes). Monitor W&B for the learning curve.

**Step 4: Verify training outputs**

After training completes, verify:

```bash
# Check that model files were saved
ls experiments/executor/run_*/

# Expected output:
# model.zip
# vec_normalize.pkl
```

```python
# Quick load verification
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.executor.env_factory import create_vec_env

# Load model
model = PPO.load("experiments/executor/run_<timestamp>/model.zip")
print(f"Policy architecture: {model.policy}")
print(f"Observation space: {model.observation_space}")
print(f"Action space: {model.action_space}")

# Count parameters
total_params = sum(p.numel() for p in model.policy.parameters())
print(f"Total parameters: {total_params}")  # Should be ~15-25K
```

**Step 5: Commit**

```bash
git add src/executor/env_factory.py src/executor/train.py
git commit -m "P2-T2: PPO training pipeline with W&B logging and VecNormalize"
```

### Acceptance Criteria

1. `src/executor/env_factory.py` creates vectorized environments compatible with SB3
2. `src/executor/train.py` trains a PPO agent and saves `model.zip` and `vec_normalize.pkl`
3. Training runs to completion without errors for 500K timesteps
4. W&B dashboard shows learning curves (mean episode reward, episode length, timesteps)
5. Saved model can be loaded with `PPO.load()` without errors
6. VecNormalize statistics are saved and can be reloaded
7. Policy architecture is 2×64 MLP with separate actor/critic networks
8. Total parameter count is in the ~15-25K range
9. Training uses 8 parallel environments via `DummyVecEnv`

### Files to Create

- `src/executor/env_factory.py`
- `src/executor/train.py`

### Files to Modify

- None

### Human Checkpoint

- Review W&B training curves — mean episode reward should trend upward
- Verify the model parameter count is reasonable (~15-25K)
- Confirm `VecNormalize` stats were saved alongside the model
- Check that training completed without NaN rewards or divergence

---

## P2-T3: Policy Distribution Extraction and Validation

**Estimated time:** ~2.5 hours
**Dependencies:** P2-T2 (trained PPO model must exist)

### Context

The **Consistency Gate (C-Gate)** is the central coordination mechanism of the Robust Trinity. It computes the **divergence** Δ_t = 1 - π_RL(d_LLM | s_t), which measures how much probability the Executor's policy assigns to the Analyst's chosen action:
1. `π_RL` — the Executor's (PPO) softmax distribution over {flat, long, short}
2. `d_LLM` — the Analyst's discrete decision (one of {flat, long, short})

For this to work, we need a function that takes a trained PPO model and an observation, and returns the full softmax probability distribution `π_RL ∈ Δ³` (a 3-element vector that sums to 1). This is NOT the same as just getting the argmax action — we need the actual probabilities.

SB3's PPO policy uses a `CategoricalDistribution` internally. The extraction process is:
1. Pass the observation through the policy network to get action logits
2. Apply softmax to convert logits to probabilities
3. Return the probability vector

This function **must work with a frozen model** (no gradient computation needed) and **must handle VecNormalize** (observations must be normalized the same way they were during training).

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Create the `get_policy_distribution()` function, validate it against the model's actual behavior, and write comprehensive tests.

### Detailed Instructions

**Step 1: Create `src/executor/policy.py`**

```python
"""Policy distribution extraction for the Executor agent.

This module provides the critical interface between the PPO Executor
and the Consistency Gate (C-Gate). The C-Gate needs the full softmax
distribution over {flat, long, short}, not just the argmax action.

The function `get_policy_distribution(model, obs)` extracts π_RL,
which is used to compute Δ_t = 1 - π_RL(d_LLM | s_t) at the C-Gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.executor.env_factory import create_vec_env


def get_policy_distribution(
    model: PPO,
    obs: np.ndarray,
    vec_normalize: VecNormalize | None = None,
) -> np.ndarray:
    """Extract the full softmax policy distribution for a given observation.

    This is the primary interface used by the C-Gate to obtain π_RL.

    Args:
        model: Trained (frozen) PPO model.
        obs: Raw observation vector of shape (obs_dim,) or (1, obs_dim).
            If vec_normalize is provided, this should be the RAW
            (un-normalized) observation.
        vec_normalize: Optional VecNormalize wrapper. If provided,
            the observation will be normalized before passing to
            the policy network.

    Returns:
        np.ndarray of shape (3,) — probability distribution over
        {flat, long, short} that sums to 1.0.
    """
    # Ensure obs is 2D: (1, obs_dim)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    obs = obs.astype(np.float32)

    # Apply VecNormalize if available
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs)

    # Convert to torch tensor
    obs_tensor = torch.as_tensor(obs, device=model.device)

    # Extract distribution from the policy network
    with torch.no_grad():
        # Get the action distribution from the policy
        dist = model.policy.get_distribution(obs_tensor)
        # For Categorical distribution, get probabilities
        probs = dist.distribution.probs.cpu().numpy()

    # Return as flat array
    return probs.squeeze()


def get_policy_distribution_batch(
    model: PPO,
    obs_batch: np.ndarray,
    vec_normalize: VecNormalize | None = None,
) -> np.ndarray:
    """Extract policy distributions for a batch of observations.

    Args:
        model: Trained (frozen) PPO model.
        obs_batch: Observations of shape (batch_size, obs_dim).
        vec_normalize: Optional VecNormalize wrapper.

    Returns:
        np.ndarray of shape (batch_size, 3) — one distribution per observation.
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.reshape(1, -1)

    obs_batch = obs_batch.astype(np.float32)

    if vec_normalize is not None:
        obs_batch = vec_normalize.normalize_obs(obs_batch)

    obs_tensor = torch.as_tensor(obs_batch, device=model.device)

    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()

    return probs


def load_executor(
    model_dir: str | Path,
) -> tuple[PPO, VecNormalize | None]:
    """Load a trained Executor model and its VecNormalize stats.

    Args:
        model_dir: Directory containing model.zip and vec_normalize.pkl.

    Returns:
        Tuple of (PPO model, VecNormalize or None).
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.zip"
    vec_norm_path = model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    model = PPO.load(str(model_path))

    # Load VecNormalize if it exists
    vec_normalize = None
    if vec_norm_path.exists():
        # Create a dummy env to load VecNormalize into
        env_fns = create_vec_env(n_envs=1, split="train")
        dummy_env = DummyVecEnv(env_fns)
        vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_env)
        vec_normalize.training = False  # Freeze normalization stats
        vec_normalize.norm_reward = False  # Don't normalize rewards at inference

    return model, vec_normalize
```

**Step 2: Create tests `tests/test_policy.py`**

```python
"""Tests for policy distribution extraction."""

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env import TradingEnv
from src.executor.policy import (
    get_policy_distribution,
    get_policy_distribution_batch,
)
from src.utils.seed import set_global_seed


@pytest.fixture
def dummy_model():
    """Create a small PPO model for testing (untrained)."""
    set_global_seed(42)
    T = 200
    n_features = 14
    features = np.random.randn(T, n_features).astype(np.float32)
    prices = 100 + np.cumsum(np.random.randn(T) * 0.5)
    prices = np.maximum(prices, 10)

    def _make():
        env = TradingEnv(
            features=features,
            prices=prices,
            lookback_window=30,
            random_start=False,
        )
        return env

    vec_env = DummyVecEnv([_make])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=42,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )

    # Run a few steps so VecNormalize has stats
    obs = vec_env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = vec_env.step(action)

    return model, vec_env


class TestGetPolicyDistribution:
    def test_output_shape(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist = get_policy_distribution(model, obs, vec_normalize=vec_env)
        assert dist.shape == (3,), f"Expected shape (3,), got {dist.shape}"

    def test_sums_to_one(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist = get_policy_distribution(model, obs, vec_normalize=vec_env)
        assert np.isclose(dist.sum(), 1.0, atol=1e-5), f"Sum = {dist.sum()}"

    def test_all_positive(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist = get_policy_distribution(model, obs, vec_normalize=vec_env)
        assert np.all(dist >= 0), f"Negative probabilities: {dist}"

    def test_no_nans(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist = get_policy_distribution(model, obs, vec_normalize=vec_env)
        assert not np.any(np.isnan(dist)), f"NaN in distribution: {dist}"

    def test_1d_and_2d_input(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist_1d = get_policy_distribution(model, obs, vec_normalize=vec_env)
        dist_2d = get_policy_distribution(model, obs.reshape(1, -1), vec_normalize=vec_env)
        np.testing.assert_array_almost_equal(dist_1d, dist_2d)

    def test_different_obs_give_different_dist(self, dummy_model):
        model, vec_env = dummy_model
        obs1 = vec_env.get_original_obs().squeeze()
        obs2 = obs1 + np.random.randn(*obs1.shape).astype(np.float32) * 10
        dist1 = get_policy_distribution(model, obs1, vec_normalize=vec_env)
        dist2 = get_policy_distribution(model, obs2, vec_normalize=vec_env)
        # Distributions should differ for different observations
        assert not np.allclose(dist1, dist2, atol=1e-3), (
            "Identical distributions for different observations"
        )

    def test_deterministic_output(self, dummy_model):
        """Same input should give same output (no stochastic sampling)."""
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        dist1 = get_policy_distribution(model, obs, vec_normalize=vec_env)
        dist2 = get_policy_distribution(model, obs, vec_normalize=vec_env)
        np.testing.assert_array_almost_equal(dist1, dist2)


class TestGetPolicyDistributionBatch:
    def test_batch_shape(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        batch = np.stack([obs, obs + 0.1, obs - 0.1])
        dists = get_policy_distribution_batch(model, batch, vec_normalize=vec_env)
        assert dists.shape == (3, 3), f"Expected (3, 3), got {dists.shape}"

    def test_batch_rows_sum_to_one(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        batch = np.stack([obs, obs + 0.5, obs - 0.5])
        dists = get_policy_distribution_batch(model, batch, vec_normalize=vec_env)
        for i in range(dists.shape[0]):
            assert np.isclose(dists[i].sum(), 1.0, atol=1e-5)

    def test_batch_consistent_with_single(self, dummy_model):
        model, vec_env = dummy_model
        obs = vec_env.get_original_obs().squeeze()
        single = get_policy_distribution(model, obs, vec_normalize=vec_env)
        batch = get_policy_distribution_batch(
            model, obs.reshape(1, -1), vec_normalize=vec_env
        )
        np.testing.assert_array_almost_equal(single, batch.squeeze())
```

**Step 3: Run tests**

```bash
pytest tests/test_policy.py -v
```

**Step 4: Integration sanity check**

Run a quick integration test to make sure the distribution extraction works with a real trained model:

```python
# Quick integration check (run interactively or as a script)
from src.executor.policy import load_executor, get_policy_distribution
from src.executor.env_factory import create_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the model from P2-T2
model, vec_norm = load_executor("experiments/executor/run_<latest>/")

# Create a fresh env to get a real observation
env_fns = create_vec_env(n_envs=1, split="train")
env = DummyVecEnv(env_fns)
obs = env.reset()

# Get distribution
dist = get_policy_distribution(model, obs.squeeze(), vec_normalize=vec_norm)
print(f"Policy distribution: flat={dist[0]:.4f}, long={dist[1]:.4f}, short={dist[2]:.4f}")
print(f"Sum: {dist.sum():.6f}")
print(f"Argmax action: {['flat', 'long', 'short'][dist.argmax()]}")
```

**Step 5: Commit**

```bash
git add src/executor/policy.py tests/test_policy.py
git commit -m "P2-T3: policy distribution extraction for C-Gate divergence computation"
```

### Acceptance Criteria

1. `get_policy_distribution(model, obs)` returns a `np.ndarray` of shape `(3,)` that sums to 1.0
2. All probabilities are non-negative and non-NaN
3. Same observation always produces the same distribution (deterministic)
4. Different observations produce different distributions
5. `get_policy_distribution_batch()` works for batched inputs with consistent results
6. `load_executor()` correctly loads both the model and VecNormalize stats
7. VecNormalize is set to inference mode (training=False, norm_reward=False) after loading
8. All tests in `tests/test_policy.py` pass

### Files to Create

- `src/executor/policy.py`
- `tests/test_policy.py`

### Files to Modify

- None

### Human Checkpoint

- Verify that the distribution output makes intuitive sense (not degenerate, not uniform)
- Confirm that VecNormalize is correctly loaded and applied at inference time
- Check that the argmax of the distribution matches `model.predict(obs, deterministic=True)`
- Review the `load_executor` function for correct model loading patterns

---

## P2-T4: Hyperparameter Tuning and Best Model Selection

**Estimated time:** ~3-4 hours (mostly wall-clock time for sweep runs)
**Dependencies:** P2-T2 (training pipeline), P2-T3 (policy distribution extraction for evaluation)

### Context

The initial PPO training in P2-T2 used default hyperparameters. For a rigorous thesis, we need to show that we explored the hyperparameter space systematically and selected the best configuration based on validation performance.

We use **W&B Sweeps** to automate this process. The sweep explores ~10-12 configurations using Bayesian optimization over the five most impactful hyperparameters. The remaining hyperparameters are fixed at established defaults from the PPO literature (Schulman et al., 2017). The objective metric is the mean episode reward on the **validation set** (a held-out portion of the training data, not the test set).

The training data spans Jan 2023 – Jun 2024. For tuning:
- **Train split:** Jan 2023 – Dec 2023
- **Validation split:** Jan 2024 – Jun 2024
- **Test split:** Jul 2024 – Dec 2024 (never used during tuning)

Hyperparameters to sweep (5 — selected for highest impact on financial RL):
- `learning_rate`: log-uniform in [1e-5, 1e-3] — most impactful for any neural network
- `n_steps`: choice of [1024, 2048, 4096] — controls bias-variance of policy gradient estimates
- `gamma`: uniform in [0.95, 0.999] — discount horizon matters significantly for trading
- `ent_coef`: log-uniform in [0.001, 0.05] — controls exploration in discrete action spaces
- `dsr_eta`: log-uniform in [0.001, 0.05] — reward adaptation rate unique to this environment

Hyperparameters fixed at defaults (4 — well-studied values with marginal tuning benefit):
- `batch_size`: 64
- `n_epochs`: 10
- `gae_lambda`: 0.95
- `clip_range`: 0.2

**Wall-clock budget:** Hard cap at 3 hours. If the sweep agent is still running, stop it and select the best result so far.

After the sweep completes, we select the best model, copy it to `experiments/executor/best_model/`, and freeze it for all downstream phases.

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Create the W&B sweep configuration, run the hyperparameter sweep, select the best model, and save it as the frozen Executor for all subsequent phases.

### Detailed Instructions

**Step 1: Create sweep configuration `configs/executor_sweep.yaml`**

```yaml
# W&B Sweep configuration for PPO Executor hyperparameter tuning
# Run with: wandb sweep configs/executor_sweep.yaml
# Then: wandb agent <sweep_id>
#
# We sweep over the 5 most impactful hyperparameters for financial RL.
# The remaining 4 (batch_size, n_epochs, gae_lambda, clip_range) are
# fixed at well-established defaults (Schulman et al., 2017).

program: src/executor/sweep_train.py
method: bayes
metric:
  name: val/mean_episode_reward
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  n_steps:
    values: [1024, 2048, 4096]
  gamma:
    distribution: uniform
    min: 0.95
    max: 0.999
  ent_coef:
    distribution: log_uniform_values
    min: 0.001
    max: 0.05
  dsr_eta:
    distribution: log_uniform_values
    min: 0.001
    max: 0.05
  # Fixed hyperparameters (not swept)
  batch_size:
    value: 64
  n_epochs:
    value: 10
  gae_lambda:
    value: 0.95
  clip_range:
    value: 0.2
  total_timesteps:
    value: 500000
  n_envs:
    value: 8
  lookback_window:
    value: 30
  seed:
    value: 42

run_cap: 12  # Hard cap on number of runs
```

**Step 2: Create sweep training script `src/executor/sweep_train.py`**

```python
"""Sweep-compatible training script for W&B hyperparameter tuning.

This script is called by `wandb agent` during a sweep. It reads
hyperparameters from wandb.config, trains a model, evaluates on
validation data, and logs the results.

Usage:
    wandb sweep configs/executor_sweep.yaml
    wandb agent <sweep_id>
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb

from src.executor.env_factory import create_vec_env
from src.executor.policy import get_policy_distribution
from src.utils.seed import set_global_seed


def evaluate_on_split(
    model: PPO,
    vec_normalize: VecNormalize,
    split: str = "val",
    n_episodes: int = 10,
    lookback_window: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a trained model on a data split.

    Args:
        model: Trained PPO model.
        vec_normalize: VecNormalize with frozen stats from training.
        split: Data split to evaluate on ("val" or "test").
        n_episodes: Number of evaluation episodes.
        lookback_window: Observation lookback window.
        seed: Random seed.

    Returns:
        Dict of evaluation metrics.
    """
    env_fns = create_vec_env(
        n_envs=1,
        split=split,
        lookback_window=lookback_window,
        seed=seed,
        random_start=False,
    )
    eval_env = DummyVecEnv(env_fns)

    # Clone VecNormalize with frozen stats
    eval_vec = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_vec.obs_rms = vec_normalize.obs_rms
    eval_vec.ret_rms = vec_normalize.ret_rms
    eval_vec.training = False

    episode_rewards = []
    episode_lengths = []
    all_distributions = []

    for ep in range(n_episodes):
        obs = eval_vec.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Also collect distribution for analysis
            dist = get_policy_distribution(model, obs.squeeze(), vec_normalize=eval_vec)
            all_distributions.append(dist)

            obs, reward, done, info = eval_vec.step(action)
            total_reward += reward[0]
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    eval_vec.close()

    # Compute distribution statistics
    all_dists = np.array(all_distributions)
    entropy = -np.sum(all_dists * np.log(all_dists + 1e-10), axis=1)

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_policy_entropy": float(np.mean(entropy)),
        "mean_flat_prob": float(np.mean(all_dists[:, 0])),
        "mean_long_prob": float(np.mean(all_dists[:, 1])),
        "mean_short_prob": float(np.mean(all_dists[:, 2])),
    }


def sweep_train():
    """Single sweep run — called by wandb agent."""
    run = wandb.init()
    config = wandb.config

    set_global_seed(config.seed)

    # Create training env
    env_fns = create_vec_env(
        n_envs=config.n_envs,
        split="train",
        lookback_window=config.lookback_window,
        seed=config.seed,
        dsr_eta=config.dsr_eta,
    )
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create model with sweep hyperparameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=0,
        seed=config.seed,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
    )

    # Train
    model.learn(total_timesteps=config.total_timesteps)

    # Evaluate on validation split
    val_metrics = evaluate_on_split(
        model, vec_env, split="val",
        lookback_window=config.lookback_window,
        seed=config.seed,
    )

    # Log validation metrics
    for key, value in val_metrics.items():
        wandb.log({f"val/{key}": value})

    # Save model for this run
    save_dir = Path("experiments") / "executor" / "sweeps" / run.id
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "model.zip"))
    vec_env.save(str(save_dir / "vec_normalize.pkl"))

    vec_env.close()
    wandb.finish()


if __name__ == "__main__":
    sweep_train()
```

**Step 3: Create the best model selection script `src/executor/select_best.py`**

```python
"""Select the best model from a W&B sweep and save as the frozen Executor.

Usage:
    python -m src.executor.select_best --sweep-id <sweep_id>

This queries the W&B API for the best run in a sweep, copies its
model and VecNormalize stats to experiments/executor/best_model/,
and logs the selection metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import wandb


def select_best_model(sweep_id: str, project: str = "robust-trinity") -> Path:
    """Select the best model from a completed sweep.

    Args:
        sweep_id: W&B sweep ID (e.g., "abc123").
        project: W&B project name.

    Returns:
        Path to the best model directory.
    """
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")

    # Get best run by validation reward
    best_run = sweep.best_run()
    print(f"Best run: {best_run.id}")
    print(f"Best val/mean_episode_reward: {best_run.summary.get('val/mean_episode_reward', 'N/A')}")
    print(f"Config: {json.dumps(dict(best_run.config), indent=2)}")

    # Source and destination paths
    src_dir = Path("experiments") / "executor" / "sweeps" / best_run.id
    dst_dir = Path("experiments") / "executor" / "best_model"

    if not src_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {src_dir}. "
            "Make sure sweep runs saved models locally."
        )

    # Copy to best_model directory
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # Save metadata
    metadata = {
        "sweep_id": sweep_id,
        "run_id": best_run.id,
        "val_mean_episode_reward": best_run.summary.get("val/mean_episode_reward"),
        "val_mean_policy_entropy": best_run.summary.get("val/mean_policy_entropy"),
        "config": dict(best_run.config),
    }
    with open(dst_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model saved to: {dst_dir}")
    print(f"Files: {list(dst_dir.iterdir())}")

    return dst_dir


def main():
    parser = argparse.ArgumentParser(description="Select best model from W&B sweep")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument("--project", default="robust-trinity", help="W&B project name")
    args = parser.parse_args()

    select_best_model(args.sweep_id, args.project)


if __name__ == "__main__":
    main()
```

**Step 4: Run the sweep**

```bash
# Initialize the sweep
wandb sweep configs/executor_sweep.yaml

# Launch agent(s) — the sweep ID is printed by the previous command
wandb agent <your-entity>/robust-trinity/<sweep_id>
```

The sweep will run ~10-12 configurations. Each run trains for 500K timesteps and evaluates on validation data. With the `run_cap: 12` setting and a 3-hour wall-clock budget, the sweep is bounded.

**Step 5: Select the best model**

```bash
python -m src.executor.select_best --sweep-id <sweep_id>

# Verify the best model was saved
ls experiments/executor/best_model/
# Expected: model.zip  vec_normalize.pkl  metadata.json
```

**Step 6: Commit**

```bash
git add configs/executor_sweep.yaml src/executor/sweep_train.py src/executor/select_best.py
git add experiments/executor/best_model/metadata.json
git commit -m "P2-T4: hyperparameter tuning via W&B sweeps and best model selection"
```

Note: Do NOT commit `model.zip` or `vec_normalize.pkl` to git (they should be in `.gitignore`). Only commit `metadata.json` which records which sweep run produced the best model.

### Acceptance Criteria

1. `configs/executor_sweep.yaml` defines a valid W&B sweep over 5 hyperparameters (lr, n_steps, gamma, ent_coef, dsr_eta) with 4 fixed at defaults
2. `src/executor/sweep_train.py` trains and evaluates a model per sweep run
3. Sweep runs log `val/mean_episode_reward` as the optimization metric
4. At least 10 sweep runs complete without errors (hard cap at 12)
5. `src/executor/select_best.py` correctly identifies and copies the best model
6. `experiments/executor/best_model/` contains `model.zip`, `vec_normalize.pkl`, and `metadata.json`
7. `metadata.json` records the sweep ID, run ID, config, and validation metrics
8. The best model achieves meaningfully better validation reward than the initial default run

### Files to Create

- `configs/executor_sweep.yaml`
- `src/executor/sweep_train.py`
- `src/executor/select_best.py`

### Files to Modify

- None

### Human Checkpoint

- Review the W&B sweep dashboard — check for convergence across the 10-12 runs
- Verify that the Bayesian optimization is exploring the 5-dim space effectively
- Confirm the best model's validation reward is significantly above random policy baseline
- Check that the best model's policy entropy is reasonable (not degenerate — should not be near 0 or near log(3))
- Approve the selected model before proceeding to the final evaluation (P2-T5)
- **Thesis note:** The 4 fixed hyperparameters (batch_size=64, n_epochs=10, gae_lambda=0.95, clip_range=0.2) should be cited as standard PPO defaults per Schulman et al. (2017)

---

## P2-T5: Out-of-Sample Evaluation of Frozen Policy

**Estimated time:** ~3 hours
**Dependencies:** P2-T4 (best model must exist at `experiments/executor/best_model/`), P2-T3 (policy distribution extraction)

### Context

The best PPO Executor model has been selected via hyperparameter tuning on validation data (P2-T4). Before it can be used in the Robust Trinity system, we must evaluate it on **truly out-of-sample test data** (Jul – Dec 2024) that was never seen during training or tuning.

This evaluation serves multiple purposes:
1. **Thesis metric:** The test-set Sharpe ratio and other metrics are reported in the thesis
2. **Sanity check:** Confirms the model generalizes and doesn't just overfit to training data
3. **Baseline for C-Gate:** The standalone Executor performance is the baseline against which the full Robust Trinity (with C-Gate, Analyst, Guardian) is compared
4. **Distribution analysis:** Characterizes the policy's behavior distribution on unseen data

The evaluation must be deterministic (using `model.predict(obs, deterministic=True)`) and must correctly apply VecNormalize statistics from training.

Key metrics to compute:
- **Sharpe ratio** (annualized, assuming 252 trading days)
- **Total return** (cumulative)
- **Maximum drawdown**
- **Win rate** (fraction of positive-return trades)
- **Average trade duration**
- **Action distribution** (fraction of flat/long/short decisions)
- **Policy entropy** (mean entropy of the softmax distribution)

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Evaluate the frozen best Executor model on the test set, compute comprehensive trading metrics, generate performance plots, and save all results.

### Detailed Instructions

**Step 1: Create `src/executor/evaluate.py`**

```python
"""Out-of-sample evaluation of the frozen Executor policy.

Usage:
    python -m src.executor.evaluate

Loads the best model from experiments/executor/best_model/,
evaluates on the test split (Jul-Dec 2024), and saves results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env_factory import create_vec_env
from src.executor.policy import (
    get_policy_distribution,
    load_executor,
)
from src.utils.seed import set_global_seed
from src.utils.logging import init_wandb, log_metrics, finish_wandb


def compute_sharpe_ratio(returns: np.ndarray, annualize: bool = True) -> float:
    """Compute the Sharpe ratio from a series of returns.

    Args:
        returns: Array of periodic returns.
        annualize: If True, annualize assuming 252 trading days.

    Returns:
        Sharpe ratio (float). Returns 0.0 if std is zero.
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    sr = np.mean(returns) / np.std(returns)
    if annualize:
        sr *= np.sqrt(252)
    return float(sr)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Compute maximum drawdown from cumulative returns.

    Args:
        cumulative_returns: Array of cumulative returns (1 + r_1)(1 + r_2)...

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 = 15% drawdown).
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1e-10)
    return float(np.max(drawdown))


def evaluate_executor(
    model_dir: str | Path = "experiments/executor/best_model",
    split: str = "test",
    seed: int = 42,
    save_results: bool = True,
) -> dict:
    """Run full evaluation of the frozen Executor policy.

    Args:
        model_dir: Path to directory with model.zip and vec_normalize.pkl.
        split: Data split to evaluate on.
        seed: Random seed.
        save_results: If True, save results to disk.

    Returns:
        Dictionary of evaluation metrics and data.
    """
    set_global_seed(seed)

    # Load model
    model, vec_normalize = load_executor(model_dir)
    print(f"Loaded model from {model_dir}")

    # Create evaluation environment (single env, no random start)
    env_fns = create_vec_env(
        n_envs=1,
        split=split,
        random_start=False,
        seed=seed,
    )
    eval_env = DummyVecEnv(env_fns)

    # Set up VecNormalize for evaluation
    if vec_normalize is not None:
        eval_vec = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_vec.obs_rms = vec_normalize.obs_rms
        eval_vec.ret_rms = vec_normalize.ret_rms
        eval_vec.training = False
    else:
        eval_vec = eval_env

    # Run evaluation
    obs = eval_vec.reset()
    done = False

    # Collect data
    actions = []
    rewards = []
    portfolio_returns = []
    positions = []
    prices = []
    distributions = []
    trades = []

    step = 0
    while not done:
        # Get action (deterministic)
        action, _ = model.predict(obs, deterministic=True)

        # Get policy distribution
        dist = get_policy_distribution(
            model, obs.squeeze(),
            vec_normalize=eval_vec if vec_normalize is not None else None,
        )
        distributions.append(dist)

        # Step
        obs, reward, done, info = eval_vec.step(action)

        # Record data
        actions.append(int(action[0]))
        rewards.append(float(reward[0]))
        if isinstance(info, list) and len(info) > 0:
            portfolio_returns.append(info[0].get("portfolio_return", 0.0))
            positions.append(info[0].get("position", 0.0))
            prices.append(info[0].get("price", 0.0))
            trades.append(info[0].get("trade", 0.0))
        else:
            portfolio_returns.append(0.0)
            positions.append(0.0)
            prices.append(0.0)
            trades.append(0.0)

        step += 1

    eval_vec.close()

    # Convert to arrays
    actions = np.array(actions)
    rewards = np.array(rewards)
    portfolio_returns = np.array(portfolio_returns)
    positions = np.array(positions)
    prices = np.array(prices)
    distributions = np.array(distributions)
    trades = np.array(trades)

    # Compute metrics
    cumulative_returns = np.cumprod(1 + portfolio_returns)

    sharpe = compute_sharpe_ratio(portfolio_returns)
    total_return = float(cumulative_returns[-1] - 1) if len(cumulative_returns) > 0 else 0.0
    max_dd = compute_max_drawdown(cumulative_returns)

    # Win rate (fraction of positive portfolio returns on trading days)
    trading_returns = portfolio_returns[trades != 0]
    win_rate = float(np.mean(trading_returns > 0)) if len(trading_returns) > 0 else 0.0

    # Action distribution
    action_counts = np.bincount(actions, minlength=3)
    action_fracs = action_counts / len(actions)

    # Policy entropy
    entropy = -np.sum(distributions * np.log(distributions + 1e-10), axis=1)
    mean_entropy = float(np.mean(entropy))

    # Trade statistics
    position_changes = np.diff(np.concatenate([[0], positions]))
    n_trades = int(np.sum(position_changes != 0))
    trade_durations = []
    current_duration = 0
    for pc in position_changes:
        if pc != 0:
            if current_duration > 0:
                trade_durations.append(current_duration)
            current_duration = 1
        else:
            current_duration += 1
    if current_duration > 0:
        trade_durations.append(current_duration)
    avg_trade_duration = float(np.mean(trade_durations)) if trade_durations else 0.0

    metrics = {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "avg_trade_duration": avg_trade_duration,
        "n_steps": step,
        "action_frac_flat": float(action_fracs[0]),
        "action_frac_long": float(action_fracs[1]),
        "action_frac_short": float(action_fracs[2]),
        "mean_policy_entropy": mean_entropy,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }

    print("\n=== Executor Test-Set Evaluation ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if save_results:
        # Save results
        results_dir = Path("experiments") / "executor" / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        with open(results_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save detailed data as CSV
        df = pd.DataFrame({
            "action": actions,
            "reward": rewards,
            "portfolio_return": portfolio_returns,
            "position": positions,
            "price": prices,
            "dist_flat": distributions[:, 0],
            "dist_long": distributions[:, 1],
            "dist_short": distributions[:, 2],
            "entropy": entropy,
        })
        df.to_csv(results_dir / "test_trajectory.csv", index=False)

        # Generate plots
        _plot_evaluation(
            cumulative_returns=cumulative_returns,
            portfolio_returns=portfolio_returns,
            positions=positions,
            distributions=distributions,
            entropy=entropy,
            prices=prices,
            save_dir=results_dir,
        )

        print(f"\nResults saved to {results_dir}")

    return metrics


def _plot_evaluation(
    cumulative_returns: np.ndarray,
    portfolio_returns: np.ndarray,
    positions: np.ndarray,
    distributions: np.ndarray,
    entropy: np.ndarray,
    prices: np.ndarray,
    save_dir: Path,
) -> None:
    """Generate evaluation plots."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Cumulative returns
    axes[0].plot(cumulative_returns, label="Executor", color="blue")
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title("Executor Policy — Test Set Performance (Jul-Dec 2024)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Position over time
    axes[1].plot(positions, color="green", alpha=0.7)
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Short", "Flat", "Long"])
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Policy distribution over time
    axes[2].stackplot(
        range(len(distributions)),
        distributions[:, 0],
        distributions[:, 1],
        distributions[:, 2],
        labels=["Flat", "Long", "Short"],
        colors=["gray", "green", "red"],
        alpha=0.7,
    )
    axes[2].set_ylabel("Policy Distribution")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Policy entropy
    axes[3].plot(entropy, color="purple", alpha=0.7)
    axes[3].axhline(
        y=np.log(3), color="gray", linestyle="--",
        alpha=0.5, label=f"Max entropy (ln3={np.log(3):.3f})"
    )
    axes[3].set_ylabel("Policy Entropy")
    axes[3].set_xlabel("Step")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "test_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Additional plot: return distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(portfolio_returns, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Portfolio Returns (Test Set)")
    plt.tight_layout()
    plt.savefig(save_dir / "return_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen Executor policy")
    parser.add_argument(
        "--model-dir",
        default="experiments/executor/best_model",
        help="Path to model directory",
    )
    parser.add_argument("--split", default="test", help="Data split to evaluate on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Initialize W&B for logging
    init_wandb(phase="P2", component="executor-eval", config=vars(args))

    metrics = evaluate_executor(
        model_dir=args.model_dir,
        split=args.split,
        seed=args.seed,
    )

    # Log to W&B
    for key, value in metrics.items():
        log_metrics({f"test/{key}": value})

    finish_wandb()


if __name__ == "__main__":
    main()
```

**Step 2: Create tests `tests/test_evaluate.py`**

```python
"""Tests for Executor evaluation utilities."""

import numpy as np
import pytest

from src.executor.evaluate import compute_sharpe_ratio, compute_max_drawdown


class TestComputeSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        sr = compute_sharpe_ratio(returns, annualize=False)
        assert sr > 0

    def test_negative_returns(self):
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.005])
        sr = compute_sharpe_ratio(returns, annualize=False)
        assert sr < 0

    def test_zero_std(self):
        returns = np.array([0.01, 0.01, 0.01])
        sr = compute_sharpe_ratio(returns, annualize=False)
        assert sr == 0.0

    def test_empty_returns(self):
        sr = compute_sharpe_ratio(np.array([]), annualize=False)
        assert sr == 0.0

    def test_annualization(self):
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        sr_daily = compute_sharpe_ratio(returns, annualize=False)
        sr_annual = compute_sharpe_ratio(returns, annualize=True)
        assert np.isclose(sr_annual, sr_daily * np.sqrt(252))


class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        cumret = np.array([1.0, 1.01, 1.02, 1.03])
        mdd = compute_max_drawdown(cumret)
        assert np.isclose(mdd, 0.0, atol=1e-6)

    def test_full_drawdown(self):
        cumret = np.array([1.0, 0.5, 0.25])
        mdd = compute_max_drawdown(cumret)
        assert mdd > 0.7  # 75% drawdown

    def test_recovery(self):
        cumret = np.array([1.0, 1.1, 0.9, 1.0, 1.2])
        mdd = compute_max_drawdown(cumret)
        expected = (1.1 - 0.9) / 1.1
        assert np.isclose(mdd, expected, atol=1e-3)
```

**Step 3: Run evaluation**

```bash
# Run evaluation on test set
python -m src.executor.evaluate

# Run tests
pytest tests/test_evaluate.py -v
```

**Step 4: Review results**

```bash
# View metrics
cat experiments/executor/evaluation/test_metrics.json

# Check plots
open experiments/executor/evaluation/test_evaluation.png
open experiments/executor/evaluation/return_distribution.png
```

**Step 5: Commit**

```bash
git add src/executor/evaluate.py tests/test_evaluate.py
git add experiments/executor/evaluation/test_metrics.json
git commit -m "P2-T5: out-of-sample evaluation of frozen Executor policy on test set"
```

### Acceptance Criteria

1. Evaluation runs on test set (Jul-Dec 2024) without errors using the frozen best model
2. All metrics are computed and saved: Sharpe ratio, total return, max drawdown, win rate, trade count, avg trade duration, action distribution, policy entropy
3. `experiments/executor/evaluation/test_metrics.json` contains all metrics
4. `experiments/executor/evaluation/test_trajectory.csv` contains step-level data
5. Performance plots are generated: cumulative returns, position timeline, distribution stacked area, entropy, return histogram
6. Sharpe ratio is reported (even if negative — the key is that evaluation runs correctly)
7. Action distribution is non-degenerate (model uses at least 2 of 3 actions in non-trivial amounts)
8. Policy entropy is between 0 and ln(3) ≈ 1.099
9. Tests for metric computation utilities pass

### Files to Create

- `src/executor/evaluate.py`
- `tests/test_evaluate.py`

### Files to Modify

- None

### Human Checkpoint

- Review the test-set Sharpe ratio — is it positive? Is it reasonable?
- Check the action distribution — does the model rely too heavily on one action?
- Look at the cumulative returns plot — are there pathological patterns (flat line, extreme drawdowns)?
- Compare the policy entropy plot to the action distribution — is the model confidently making diverse decisions?
- Verify that the test metrics are reasonable for the thesis baseline
- **This is the final checkpoint for the Executor.** After this, the frozen model at `experiments/executor/best_model/` will be used in all subsequent phases (C-Gate, Guardian, ablation experiments)
