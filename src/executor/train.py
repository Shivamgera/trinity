"""PPO training script for the Executor agent.

Usage:
    python3 -m src.executor.train
    python3 -m src.executor.train --timesteps 500000 --n_envs 8
    python3 -m src.executor.train --timesteps 100000 --no-wandb  # quick test

The trained model is saved to:
    experiments/executor/run_<timestamp>/model.zip
    experiments/executor/run_<timestamp>/vec_normalize.pkl
    experiments/executor/run_<timestamp>/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env_factory import create_vec_env
from src.executor.evaluate import compute_sharpe_ratio
from src.utils.logging import finish_wandb, init_wandb, log_metrics

logger = logging.getLogger(__name__)


class WandbCallback(BaseCallback):
    """Callback that logs training metrics to W&B at regular intervals."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self._ep_rew_buffer: list[float] = []
        self._episode_returns: list[float] = []
        self._portfolio_returns: list[float] = []

    def _on_step(self) -> bool:
        # Collect episode returns and per-step portfolio returns from infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rew_buffer.append(info["episode"]["r"])
            if "portfolio_return" in info:
                self._portfolio_returns.append(info["portfolio_return"])

        if self.n_calls % self.log_freq == 0 and self._ep_rew_buffer:
            mean_ep_rew = float(np.mean(self._ep_rew_buffer[-100:]))
            metrics: dict = {
                "train/mean_episode_reward": mean_ep_rew,
                "train/timesteps": self.num_timesteps,
            }
            # Rolling Sharpe from last 2000 portfolio returns (~8 envs × 250 steps)
            if len(self._portfolio_returns) >= 50:
                recent = np.array(self._portfolio_returns[-2000:])
                metrics["train/sharpe_ratio"] = compute_sharpe_ratio(recent, annualize=True)
            log_metrics(metrics, step=self.num_timesteps)

        return True


def train(
    ticker: str = "AAPL",
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    lookback_window: int = 10,
    transaction_cost: float = 0.001,
    dsr_eta: float = 0.01,
    seed: int = 42,
    use_wandb: bool = True,
    run_dir: Path | None = None,
) -> tuple[PPO, VecNormalize, Path]:
    """Train a PPO agent on the TradingEnv.

    Args:
        ticker: Ticker symbol to train on.
        total_timesteps: Total training steps.
        n_envs: Parallel environments (DummyVecEnv).
        learning_rate: PPO learning rate.
        n_steps: Steps per environment per rollout.
        batch_size: Mini-batch size.
        n_epochs: Optimization epochs per rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        lookback_window: Observation lookback window.
        transaction_cost: Cost per unit position change.
        dsr_eta: DSR adaptation rate.
        seed: Random seed.
        use_wandb: Log metrics to W&B.
        run_dir: Output directory. Auto-generated if None.

    Returns:
        Tuple of (trained PPO model, VecNormalize wrapper, output directory).
    """
    # ------------------------------------------------------------------ output
    if run_dir is None:
        timestamp = int(time.time())
        run_dir = Path("experiments/executor") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ticker": ticker,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "lookback_window": lookback_window,
        "transaction_cost": transaction_cost,
        "dsr_eta": dsr_eta,
        "seed": seed,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----------------------------------------------------------------- logging
    if use_wandb:
        init_wandb(phase="phase2", component="executor", config=config)
        logger.info("W&B run initialized")

    # ------------------------------------------------------------- environment
    logger.info(f"Creating {n_envs} parallel training environments...")
    env_fns = create_vec_env(
        n_envs=n_envs,
        ticker=ticker,
        split="train",
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
        dsr_eta=dsr_eta,
        random_start=True,
    )
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    # --------------------------------------------------------------- PPO model
    policy_kwargs = {
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": __import__("torch.nn", fromlist=["Tanh"]).Tanh,
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    logger.info(f"PPO model has {sum(p.numel() for p in model.policy.parameters())} parameters")
    logger.info(f"Training for {total_timesteps:,} timesteps...")

    # ---------------------------------------------------------------- training
    callbacks = []
    if use_wandb:
        callbacks.append(WandbCallback(log_freq=1000))

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    # ------------------------------------------------------------------- save
    model_path = run_dir / "model.zip"
    vn_path = run_dir / "vec_normalize.pkl"

    model.save(str(model_path))
    vec_env.save(str(vn_path))

    logger.info(f"Model saved to {model_path}")
    logger.info(f"VecNormalize saved to {vn_path}")

    if use_wandb:
        finish_wandb()

    return model, vec_env, run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train PPO Executor agent")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    train(
        ticker=args.ticker,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        dsr_eta=args.dsr_eta,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
