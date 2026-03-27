"""Tests for TradingEnv and DifferentialSharpeReward.

Run with:
    python3 -m pytest tests/test_env.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.executor.env import ACTION_TO_POSITION, TradingEnv
from src.executor.rewards import DifferentialSharpeReward, LogReturnReward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_env(
    n_days: int = 100,
    n_features: int = 14,
    lookback: int = 10,
    seed: int = 42,
    random_start: bool = False,
    episode_length: int | None = None,
) -> TradingEnv:
    """Build a TradingEnv from synthetic data."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_days, n_features)).astype(np.float32)
    prices = 100.0 * np.cumprod(1 + rng.standard_normal(n_days) * 0.01)

    dates = pd.bdate_range(start="2023-01-02", periods=n_days)
    features_df = pd.DataFrame(
        features,
        index=dates,
        columns=[f"f{i}" for i in range(n_features)],
    )
    prices_series = pd.Series(prices, index=dates)

    return TradingEnv(
        features=features_df,
        prices=prices_series,
        lookback_window=lookback,
        random_start=random_start,
        episode_length=episode_length,
    )


# ---------------------------------------------------------------------------
# DifferentialSharpeReward Tests
# ---------------------------------------------------------------------------


class TestDifferentialSharpeReward:
    def test_reset_clears_state(self):
        dsr = DifferentialSharpeReward(eta=0.01)
        for _ in range(10):
            dsr.compute(0.01)
        assert dsr.A != 0.0 or dsr.B != 0.0
        dsr.reset()
        assert dsr.A == 0.0
        assert dsr.B == 0.0
        assert dsr._step == 0

    def test_bootstrap_returns_plain_return(self):
        """First two steps fall back to simple return."""
        dsr = DifferentialSharpeReward(eta=0.01)
        r1 = dsr.compute(0.05)
        r2 = dsr.compute(-0.03)
        assert r1 == pytest.approx(0.05)
        assert r2 == pytest.approx(-0.03)

    def test_zero_variance_fallback(self):
        """Constant returns → zero variance → fall back to simple return."""
        dsr = DifferentialSharpeReward(eta=0.5)
        for _ in range(20):
            reward = dsr.compute(0.0)
        # Should not raise; reward should be finite
        assert np.isfinite(reward)

    def test_positive_reward_positive_returns(self):
        """Consistently positive returns should yield positive DSR."""
        dsr = DifferentialSharpeReward(eta=0.01)
        rewards = [dsr.compute(0.01) for _ in range(50)]
        # After warmup, rewards should trend positive
        assert sum(rewards[10:]) > 0

    def test_eta_parameter(self):
        """Different eta values should produce different A/B estimates."""
        dsr_slow = DifferentialSharpeReward(eta=0.001)
        dsr_fast = DifferentialSharpeReward(eta=0.1)
        for _ in range(5):
            dsr_slow.compute(0.01)
            dsr_fast.compute(0.01)
        assert dsr_fast.A != pytest.approx(dsr_slow.A)


# ---------------------------------------------------------------------------
# TradingEnv — Spaces and Construction
# ---------------------------------------------------------------------------


class TestTradingEnvSpaces:
    def test_observation_dim(self):
        env = _make_env(n_days=100, n_features=14, lookback=10)
        obs, _ = env.reset(seed=0)
        # 10 * 14 + 3 = 143
        assert obs.shape == (143,)

    def test_observation_dtype(self):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_action_space(self):
        env = _make_env()
        assert env.action_space.n == 3

    def test_action_to_position_keys(self):
        assert set(ACTION_TO_POSITION.keys()) == {0, 1, 2}
        assert ACTION_TO_POSITION[0] == 0.0  # flat
        assert ACTION_TO_POSITION[1] == 1.0  # long
        assert ACTION_TO_POSITION[2] == -1.0  # short

    def test_observation_space_shape(self):
        env = _make_env()
        assert env.observation_space.shape == (env.obs_dim,)


# ---------------------------------------------------------------------------
# TradingEnv — reset / step
# ---------------------------------------------------------------------------


class TestTradingEnvStep:
    def test_reset_returns_obs_and_info(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)
        assert isinstance(info, dict)

    def test_step_output_types(self):
        env = _make_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (env.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_info_keys_present(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        for key in ("step", "position", "unrealized_pnl", "total_pnl", "price"):
            assert key in info, f"Missing key: {key}"

    def test_position_updates_correctly(self):
        env = _make_env(random_start=False)
        env.reset(seed=0)
        env.step(1)  # go long
        assert env._position == pytest.approx(1.0)
        env.step(0)  # go flat
        assert env._position == pytest.approx(0.0)
        env.step(2)  # go short
        assert env._position == pytest.approx(-1.0)

    def test_transaction_cost_applies_on_trade(self):
        """When position changes, info should report nonzero transaction cost."""
        env = _make_env(random_start=False)
        env.reset(seed=0)
        _, _, _, _, info = env.step(1)  # flat -> long, should cost
        assert info["transaction_cost"] > 0

    def test_no_transaction_cost_on_hold(self):
        """Holding the same position incurs no transaction cost."""
        env = _make_env(random_start=False)
        env.reset(seed=0)
        env.step(1)  # go long
        _, _, _, _, info = env.step(1)  # hold long
        assert info["transaction_cost"] == pytest.approx(0.0)

    def test_long_return_immediate_no_delay(self):
        """Going long when price rises should produce a positive portfolio
        return on the *same* step — no one-step delay."""
        # Build env with deterministic, monotonically rising prices
        n_days, n_features, lookback = 60, 14, 10
        features = np.zeros((n_days, n_features), dtype=np.float32)
        prices = np.linspace(100.0, 110.0, n_days)  # steady rise

        dates = pd.bdate_range(start="2023-01-02", periods=n_days)
        features_df = pd.DataFrame(
            features, index=dates,
            columns=[f"f{i}" for i in range(n_features)],
        )
        prices_series = pd.Series(prices, index=dates)

        env = TradingEnv(
            features=features_df,
            prices=prices_series,
            lookback_window=lookback,
            random_start=False,
            transaction_cost=0.0,  # zero TC to isolate the position P&L
            slippage=0.0,
        )
        env.reset(seed=0)

        # Step 1: go long from flat.  Price rises, so portfolio_return
        # should be positive on this very step.
        _, _, _, _, info = env.step(1)
        assert info["portfolio_return"] > 0, (
            f"Expected positive return when going long in a rising market, "
            f"got {info['portfolio_return']}"
        )

    def test_episode_does_not_exceed_data(self):
        env = _make_env(n_days=60, lookback=10, random_start=False)
        env.reset(seed=0)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(1)
            steps += 1
        assert steps > 0

    def test_episode_length_truncation(self):
        env = _make_env(n_days=100, lookback=10, random_start=False, episode_length=10)
        env.reset(seed=0)
        truncated = False
        terminated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
        assert steps <= 10


# ---------------------------------------------------------------------------
# TradingEnv — Date tracking (for C-Gate integration)
# ---------------------------------------------------------------------------


class TestTradingEnvDateTracking:
    def test_current_date_returns_string(self):
        env = _make_env()
        env.reset(seed=0)
        date = env.current_date
        assert date is not None
        assert len(date) == 10  # "YYYY-MM-DD"

    def test_date_in_info_when_dataframe(self):
        env = _make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert "date" in info

    def test_no_date_with_numpy_input(self):
        """When constructed from numpy arrays, no date tracking."""
        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((80, 14)).astype(np.float32)
        prices_np = 100.0 + np.arange(80, dtype=np.float64)
        env = TradingEnv(
            features=features_np,
            prices=prices_np,
            random_start=False,
        )
        env.reset(seed=0)
        assert env.current_date is None


# ---------------------------------------------------------------------------
# LogReturnReward Tests
# ---------------------------------------------------------------------------


class TestLogReturnReward:
    def test_positive_return_positive_reward(self):
        lr = LogReturnReward()
        assert lr.compute(0.01) > 0.0

    def test_negative_return_negative_reward(self):
        lr = LogReturnReward()
        assert lr.compute(-0.01) < 0.0

    def test_zero_return_zero_reward(self):
        lr = LogReturnReward()
        assert lr.compute(0.0) == pytest.approx(0.0)

    def test_log_transform_value(self):
        """Verify the exact log(1+r) formula."""
        lr = LogReturnReward()
        r = 0.05
        expected = float(np.log1p(r))
        assert lr.compute(r) == pytest.approx(expected)

    def test_large_negative_return(self):
        """A -50% drawdown should produce a finite negative reward."""
        lr = LogReturnReward()
        reward = lr.compute(-0.50)
        assert np.isfinite(reward)
        assert reward < 0.0

    def test_asymmetry(self):
        """log(1+r) penalises losses more than it rewards equivalent gains."""
        lr = LogReturnReward()
        gain = lr.compute(0.05)
        loss = lr.compute(-0.05)
        assert abs(loss) > abs(gain)

    def test_reset_is_noop(self):
        """reset() should not raise or change behaviour."""
        lr = LogReturnReward()
        r1 = lr.compute(0.02)
        lr.reset()
        r2 = lr.compute(0.02)
        assert r1 == pytest.approx(r2)

    def test_stateless_no_path_dependence(self):
        """Same input should always produce same output regardless of history."""
        lr = LogReturnReward()
        # Compute several values first
        lr.compute(0.10)
        lr.compute(-0.05)
        lr.compute(0.03)
        # Now compute target
        result = lr.compute(0.02)
        # Fresh instance, same input
        lr2 = LogReturnReward()
        assert lr2.compute(0.02) == pytest.approx(result)


# ---------------------------------------------------------------------------
# TradingEnv — LogReturn reward_type integration
# ---------------------------------------------------------------------------


class TestTradingEnvLogReturn:
    def test_log_return_env_produces_finite_rewards(self):
        """TradingEnv with reward_type='log_return' runs without errors."""
        env = _make_env(n_days=60, lookback=10, random_start=False)
        # Rebuild with reward_type
        rng = np.random.default_rng(42)
        features = rng.standard_normal((60, 14)).astype(np.float32)
        prices = 100.0 * np.cumprod(1 + rng.standard_normal(60) * 0.01)
        dates = pd.bdate_range(start="2023-01-02", periods=60)
        features_df = pd.DataFrame(
            features, index=dates,
            columns=[f"f{i}" for i in range(14)],
        )
        prices_series = pd.Series(prices, index=dates)

        env = TradingEnv(
            features=features_df,
            prices=prices_series,
            lookback_window=10,
            random_start=False,
            reward_type="log_return",
        )
        obs, _ = env.reset(seed=0)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(1)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            total_reward += reward
        # Should have completed without error and reward should be finite
        assert np.isfinite(total_reward)

    def test_invalid_reward_type_raises(self):
        """Unknown reward_type should raise ValueError."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((60, 14)).astype(np.float32)
        prices = 100.0 + np.arange(60, dtype=np.float64)
        dates = pd.bdate_range(start="2023-01-02", periods=60)
        features_df = pd.DataFrame(
            features, index=dates,
            columns=[f"f{i}" for i in range(14)],
        )
        prices_series = pd.Series(prices, index=dates)

        with pytest.raises(ValueError, match="Unknown reward_type"):
            TradingEnv(
                features=features_df,
                prices=prices_series,
                lookback_window=10,
                reward_type="invalid_type",
            )
