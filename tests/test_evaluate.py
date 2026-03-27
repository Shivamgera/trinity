"""Tests for Executor evaluation metrics and utilities.

Run with:
    python3 -m pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.executor.evaluate import (
    EVAL_DIR,
    compute_max_drawdown,
    compute_sharpe_ratio,
    evaluate_executor,
)


# ---------------------------------------------------------------------------
# compute_sharpe_ratio
# ---------------------------------------------------------------------------


class TestComputeSharpeRatio:
    def test_empty_returns_zero(self):
        assert compute_sharpe_ratio(np.array([])) == 0.0

    def test_zero_std_returns_zero(self):
        returns = np.full(100, 0.001)
        assert compute_sharpe_ratio(returns) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.001, scale=0.01, size=1000)
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe > 0

    def test_negative_mean_negative_sharpe(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=-0.001, scale=0.01, size=1000)
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe < 0

    def test_annualized_larger_than_raw(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.001, scale=0.01, size=1000)
        raw = compute_sharpe_ratio(returns, annualize=False)
        annualized = compute_sharpe_ratio(returns, annualize=True)
        # Annualized should be sqrt(252) ≈ 15.87x larger
        assert annualized > raw


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


class TestComputeMaxDrawdown:
    def test_empty_returns_zero(self):
        assert compute_max_drawdown(np.array([])) == 0.0

    def test_monotonically_increasing_zero_drawdown(self):
        cum_returns = np.linspace(1.0, 2.0, 100)
        dd = compute_max_drawdown(cum_returns)
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_monotonically_decreasing_large_drawdown(self):
        cum_returns = np.linspace(2.0, 1.0, 100)  # 50% drawdown
        dd = compute_max_drawdown(cum_returns)
        assert dd == pytest.approx(0.5, rel=1e-3)

    def test_drawdown_is_positive(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.01, 500)
        cum = np.cumprod(1 + returns)
        dd = compute_max_drawdown(cum)
        assert dd >= 0

    def test_known_drawdown(self):
        """Peak at step 1 (value=2), trough at step 2 (value=1) → 50% DD."""
        cum = np.array([1.0, 2.0, 1.0, 2.0])
        dd = compute_max_drawdown(cum)
        assert dd == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# evaluate_executor (integration: needs trained model)
# ---------------------------------------------------------------------------


class TestEvaluateExecutorMetricsStructure:
    """Smoke-test evaluate_executor using a tiny trained model."""

    @pytest.fixture(autouse=True)
    def setup_tiny_model(self, tmp_path, monkeypatch):
        """Train a tiny model and point EVAL_DIR to tmp_path."""
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from src.executor.env import TradingEnv

        rng = np.random.default_rng(0)
        n = 80

        def env_fn():
            features = rng.standard_normal((n, 14)).astype(np.float32)
            prices = 100.0 + np.arange(n, dtype=np.float64)
            return TradingEnv(features=features, prices=prices, random_start=False)

        vec_env = DummyVecEnv([env_fn])
        model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
        model.learn(total_timesteps=128)

        self.model_dir = tmp_path / "model"
        self.model_dir.mkdir()
        model.save(str(self.model_dir / "model.zip"))

        # Monkeypatch evaluate_executor to use synthetic env
        # (avoids disk I/O to data/)
        import src.executor.evaluate as ev_module

        original_load = ev_module.load_executor
        original_make = ev_module.make_trading_env

        monkeypatch.setattr(
            ev_module,
            "load_executor",
            lambda d: (model, None),
        )
        monkeypatch.setattr(
            ev_module,
            "make_trading_env",
            lambda **kwargs: env_fn,
        )

        # Redirect EVAL_DIR
        monkeypatch.setattr(ev_module, "EVAL_DIR", tmp_path / "eval")
        self.eval_dir = tmp_path / "eval"

    def test_returns_dict_with_required_keys(self):
        import src.executor.evaluate as ev_module

        metrics = ev_module.evaluate_executor(
            model_dir=self.model_dir,
            split="test",
            use_wandb=False,
        )
        required_keys = [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "mean_episode_reward",
            "pct_flat",
            "pct_long",
            "pct_short",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_position_fractions_sum_to_one(self):
        import src.executor.evaluate as ev_module

        metrics = ev_module.evaluate_executor(
            model_dir=self.model_dir,
            split="test",
            use_wandb=False,
        )
        pos_sum = metrics["pct_flat"] + metrics["pct_long"] + metrics["pct_short"]
        assert pos_sum == pytest.approx(1.0, abs=0.01)

    def test_evaluation_json_saved(self):
        import src.executor.evaluate as ev_module

        ev_module.evaluate_executor(
            model_dir=self.model_dir,
            split="test",
            use_wandb=False,
        )
        assert (self.eval_dir / "evaluation_results.json").exists()
        with open(self.eval_dir / "evaluation_results.json") as f:
            saved = json.load(f)
        assert "sharpe_ratio" in saved

    def test_plots_saved(self):
        import src.executor.evaluate as ev_module

        ev_module.evaluate_executor(
            model_dir=self.model_dir,
            split="test",
            use_wandb=False,
        )
        assert (self.eval_dir / "test_evaluation.png").exists()
        assert (self.eval_dir / "return_distribution.png").exists()
