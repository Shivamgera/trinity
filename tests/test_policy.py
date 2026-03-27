"""Tests for policy distribution extraction utilities.

Run with:
    python3 -m pytest tests/test_policy.py -v
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env import TradingEnv
from src.executor.policy import (
    get_policy_distribution,
    get_policy_distribution_batch,
    load_executor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_env() -> TradingEnv:
    """Build a minimal TradingEnv for testing (no file I/O)."""
    rng = np.random.default_rng(0)
    n = 80
    features = rng.standard_normal((n, 14)).astype(np.float32)
    prices = 100.0 + np.arange(n, dtype=np.float64)
    return TradingEnv(features=features, prices=prices, lookback_window=10, random_start=False)


def _make_tiny_model() -> tuple[PPO, DummyVecEnv]:
    """Train a PPO model for a few steps (enough to exist)."""
    env_fn = _make_tiny_env
    vec_env = DummyVecEnv([env_fn])
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
    )
    model.learn(total_timesteps=128)
    return model, vec_env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetPolicyDistribution:
    def setup_method(self):
        self.model, self.vec_env = _make_tiny_model()
        env = _make_tiny_env()
        obs, _ = env.reset(seed=0)
        self.obs = obs

    def test_output_shape(self):
        probs = get_policy_distribution(self.model, self.obs)
        assert probs.shape == (3,)

    def test_output_sums_to_one(self):
        probs = get_policy_distribution(self.model, self.obs)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_all_probabilities_nonnegative(self):
        probs = get_policy_distribution(self.model, self.obs)
        assert np.all(probs >= 0)

    def test_2d_input_accepted(self):
        """Passing shape (1, obs_dim) should also work."""
        obs_2d = self.obs[np.newaxis, :]
        probs = get_policy_distribution(self.model, obs_2d)
        assert probs.shape == (3,)

    def test_output_dtype_float(self):
        probs = get_policy_distribution(self.model, self.obs)
        assert probs.dtype in (np.float32, np.float64)

    def test_temperature_default_matches_one(self):
        """Default temperature (1.0) should match explicit T=1.0."""
        p_default = get_policy_distribution(self.model, self.obs)
        p_one = get_policy_distribution(self.model, self.obs, temperature=1.0)
        np.testing.assert_allclose(p_default, p_one, atol=1e-6)

    def test_low_temperature_sharpens(self):
        """Lower temperature should produce a more peaked distribution."""
        p_default = get_policy_distribution(self.model, self.obs, temperature=1.0)
        p_sharp = get_policy_distribution(self.model, self.obs, temperature=0.1)
        # Sharper distribution has higher max probability
        assert p_sharp.max() >= p_default.max()
        # Argmax should be the same action
        assert np.argmax(p_sharp) == np.argmax(p_default)

    def test_low_temperature_still_valid(self):
        """Temperature-scaled output should still be a valid distribution."""
        probs = get_policy_distribution(self.model, self.obs, temperature=0.01)
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(probs >= 0)

    def test_temperature_zero_raises(self):
        """Temperature of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            get_policy_distribution(self.model, self.obs, temperature=0.0)

    def test_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            get_policy_distribution(self.model, self.obs, temperature=-0.5)


class TestGetPolicyDistributionBatch:
    def setup_method(self):
        self.model, _ = _make_tiny_model()
        env = _make_tiny_env()
        obs, _ = env.reset(seed=0)
        self.obs = obs

    def test_batch_output_shape(self):
        obs_batch = np.stack([self.obs] * 5)
        probs = get_policy_distribution_batch(self.model, obs_batch)
        assert probs.shape == (5, 3)

    def test_batch_rows_sum_to_one(self):
        obs_batch = np.stack([self.obs] * 4)
        probs = get_policy_distribution_batch(self.model, obs_batch)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_single_vs_batch_consistent(self):
        """Single and batch should return same result for the same obs."""
        single = get_policy_distribution(self.model, self.obs)
        batch = get_policy_distribution_batch(self.model, self.obs[np.newaxis, :])
        np.testing.assert_allclose(single, batch[0], atol=1e-5)

    def test_batch_temperature_consistent_with_single(self):
        """Batch with temperature should match single with temperature."""
        T = 0.1
        single = get_policy_distribution(self.model, self.obs, temperature=T)
        batch = get_policy_distribution_batch(
            self.model, self.obs[np.newaxis, :], temperature=T
        )
        np.testing.assert_allclose(single, batch[0], atol=1e-5)

    def test_batch_temperature_zero_raises(self):
        """Batch with temperature=0 should raise ValueError."""
        obs_batch = np.stack([self.obs] * 3)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            get_policy_distribution_batch(self.model, obs_batch, temperature=0.0)


class TestLoadExecutor:
    def test_load_model_saves_and_loads(self):
        """Train a tiny model, save it, then load it back."""
        model, _ = _make_tiny_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            model.save(str(tmppath / "model.zip"))
            loaded_model, vec_normalize = load_executor(tmppath)
            assert isinstance(loaded_model, PPO)
            assert vec_normalize is None  # no vec_normalize.pkl written

    def test_load_with_vec_normalize(self):
        """Save + load with VecNormalize."""
        env_fn = _make_tiny_env
        vec_env = DummyVecEnv([env_fn])
        vn = VecNormalize(vec_env)
        model = PPO("MlpPolicy", vn, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
        model.learn(total_timesteps=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            model.save(str(tmppath / "model.zip"))
            vn.save(str(tmppath / "vec_normalize.pkl"))

            loaded_model, loaded_vn = load_executor(tmppath)
            assert isinstance(loaded_model, PPO)
            assert loaded_vn is not None
            # Should be in inference mode
            assert loaded_vn.training is False
            assert loaded_vn.norm_reward is False

    def test_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_executor(tmpdir)
