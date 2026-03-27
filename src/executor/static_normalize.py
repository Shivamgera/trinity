"""Static observation normalization for DQN training.

DQN uses an off-policy replay buffer where old experiences are re-sampled
during training.  Dynamic normalization (VecNormalize) creates a distribution
shift: observations stored early in training were normalized with different
running statistics than those seen later.  This destabilizes Q-value
estimation.

Solution: compute mean and std from the TRAINING SET ONLY using random-policy
rollouts, then apply those frozen constants to all environments (train, val,
test).  This guarantees every observation---whether freshly collected or
replayed from the buffer---lives in the same distribution.

Usage:
    # One-time: compute and save stats
    python3 -m src.executor.static_normalize

    # In training/evaluation code:
    from src.executor.static_normalize import StaticNormalizer, StaticNormWrapper
    normalizer = StaticNormalizer.load("experiments/executor_dqn/normalization_stats.json")
    env = StaticNormWrapper(env, normalizer)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.executor.env_factory import make_trading_env

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = Path("experiments/executor_dqn/normalization_stats.json")


class StaticNormalizer:
    """Applies frozen (obs - mean) / (std + eps) normalization.

    Drop-in replacement for VecNormalize's normalize_obs() method,
    but with completely static statistics that never update.
    """

    def __init__(
        self,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        clip_obs: float = 10.0,
        eps: float = 1e-8,
    ):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)
        self.clip_obs = clip_obs
        self.eps = eps

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation(s) using frozen statistics.

        Args:
            obs: Single observation (obs_dim,) or batch (B, obs_dim).

        Returns:
            Normalized and clipped observation(s), same shape as input.
        """
        normalized = (obs.astype(np.float32) - self.obs_mean) / (self.obs_std + self.eps)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save normalization statistics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "clip_obs": self.clip_obs,
            "obs_dim": len(self.obs_mean),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Normalization stats saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> StaticNormalizer:
        """Load normalization statistics from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            obs_mean=np.array(data["obs_mean"], dtype=np.float32),
            obs_std=np.array(data["obs_std"], dtype=np.float32),
            clip_obs=data.get("clip_obs", 10.0),
        )


class StaticNormWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies static normalization to observations.

    Wraps a single (non-vectorized) environment.  For use with DQN which
    requires a single env wrapped in DummyVecEnv.

    This replaces VecNormalize for DQN training.  Observations from both
    reset() and step() are normalized using frozen training-set statistics.
    """

    def __init__(self, env: gym.Env, normalizer: StaticNormalizer):
        super().__init__(env)
        self.normalizer = normalizer

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.normalizer.normalize_obs(obs), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalizer.normalize_obs(obs), reward, terminated, truncated, info


def compute_normalization_stats(
    ticker: str = "AAPL",
    n_episodes: int = 50,
    reward_type: str = "log_return",
    save_path: str | Path = DEFAULT_STATS_PATH,
) -> StaticNormalizer:
    """Collect observations from random-policy rollouts on training data.

    Runs n_episodes of the training environment with a random policy to
    collect a representative sample of observations, then computes the
    mean and std (ddof=0) across all collected observations.

    Args:
        ticker: Ticker symbol.
        n_episodes: Number of random-policy episodes to collect.
        reward_type: Reward function for the environment.
        save_path: Where to save the JSON stats file.

    Returns:
        StaticNormalizer with computed statistics.
    """
    logger.info(f"Collecting observations from {n_episodes} random-policy episodes...")

    env_fn = make_trading_env(
        ticker=ticker,
        split="train",
        random_start=True,
        reward_type=reward_type,
    )
    env = env_fn()

    all_obs: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        all_obs.append(obs)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            all_obs.append(obs)
            done = terminated or truncated

    env.close()

    obs_array = np.stack(all_obs, axis=0)  # (N, obs_dim)
    obs_mean = np.mean(obs_array, axis=0)
    obs_std = np.std(obs_array, axis=0)  # ddof=0

    logger.info(
        f"Computed stats from {len(all_obs)} observations "
        f"(obs_dim={obs_mean.shape[0]}). "
        f"Mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}], "
        f"Std range: [{obs_std.min():.6f}, {obs_std.max():.4f}]"
    )

    normalizer = StaticNormalizer(obs_mean=obs_mean, obs_std=obs_std)
    normalizer.save(save_path)
    return normalizer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    compute_normalization_stats()
