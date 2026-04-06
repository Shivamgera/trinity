"""W&B sweep training script for DQN hyperparameter search.

Invoked by the W&B agent once per sweep run.  Reads DQN-specific
hyperparameter values from wandb.config, trains with early stopping
based on validation Sharpe ratio, and reports val/sharpe_ratio as
the sweep selection metric.

Usage (launch sweep then start agent):
    wandb sweep configs/dqn_sweep.yaml
    wandb agent <sweep_id>

Or programmatically:
    python3 -m src.executor.sweep_train_dqn
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from src.executor.env_factory import make_trading_env
from src.executor.static_normalize import (
    StaticNormalizer,
    StaticNormWrapper,
    compute_normalization_stats,
    DEFAULT_STATS_PATH,
)
from src.executor.train_dqn import (
    DQNValCheckpointCallback,
    DQNWandbCallback,
    evaluate_dqn_on_split,
)

logger = logging.getLogger(__name__)


def sweep_train() -> None:
    """Single sweep run: train DQN with early stopping, report best val metrics."""
    with wandb.init() as run:
        cfg = wandb.config

        # ---- Read tuned hyperparameters from sweep config ----
        learning_rate = cfg.get("learning_rate", 1.98e-4)
        gamma = cfg.get("gamma", 0.95)
        buffer_size = cfg.get("buffer_size", 50_000)
        learning_starts = cfg.get("learning_starts", 1_000)
        exploration_fraction = cfg.get("exploration_fraction", 0.4)
        exploration_final_eps = cfg.get("exploration_final_eps", 0.05)
        target_update_interval = cfg.get("target_update_interval", 1_000)
        train_freq = cfg.get("train_freq", 4)
        gradient_steps = cfg.get("gradient_steps", 1)

        # ---- Fixed hyperparameters ----
        total_timesteps = 250_000
        batch_size = 64
        tau = 1.0
        net_arch = [64, 64]
        eval_freq = 2048
        patience = 25
        min_timesteps = 30_000
        reward_type = "log_return"
        seed = 42

        run_dir = Path("experiments/executor_dqn/sweep") / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DQN sweep run {run.id}: lr={learning_rate:.2e}, gamma={gamma:.3f}, "
            f"buffer={buffer_size}, starts={learning_starts}, "
            f"explore_frac={exploration_fraction:.2f}, "
            f"final_eps={exploration_final_eps:.3f}, "
            f"target_update={target_update_interval}, "
            f"train_freq={train_freq}, grad_steps={gradient_steps}"
        )

        # ---- Load or compute normalization stats ----
        if DEFAULT_STATS_PATH.exists():
            normalizer = StaticNormalizer.load(DEFAULT_STATS_PATH)
        else:
            normalizer = compute_normalization_stats(reward_type=reward_type)

        # ---- Build environment ----
        env_fn = make_trading_env(
            split="train",
            random_start=True,
            reward_type=reward_type,
        )
        raw_env = env_fn()
        norm_env = StaticNormWrapper(raw_env, normalizer)
        vec_env = DummyVecEnv([lambda: norm_env])

        # ---- Build DQN model ----
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": nn.Tanh,
        }

        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
        )

        # ---- Callbacks ----
        val_cb = DQNValCheckpointCallback(
            normalizer=normalizer,
            run_dir=run_dir,
            eval_freq=eval_freq,
            patience=patience,
            min_timesteps=min_timesteps,
            reward_type=reward_type,
        )
        wandb_cb = DQNWandbCallback(log_freq=1000)

        # ---- Train ----
        model.learn(
            total_timesteps=total_timesteps,
            callback=[val_cb, wandb_cb],
            progress_bar=False,
        )

        # ---- Copy best checkpoint ----
        best_dir = run_dir / "best"
        if best_dir.exists():
            shutil.copy2(best_dir / "model.zip", run_dir / "model.zip")
            if (best_dir / "normalization_stats.json").exists():
                shutil.copy2(
                    best_dir / "normalization_stats.json",
                    run_dir / "normalization_stats.json",
                )
        else:
            model.save(str(run_dir / "model.zip"))

        # ---- Report best val metrics ----
        metrics = val_cb.best_metrics
        if not metrics:
            metrics = evaluate_dqn_on_split(
                model=model,
                normalizer=normalizer,
                split="val",
                reward_type=reward_type,
            )

        wandb.log(
            {
                "val/mean_episode_reward": metrics["mean_episode_reward"],
                "val/sharpe_ratio": metrics["sharpe_ratio"],
                "val/max_drawdown": metrics["max_drawdown"],
                "val/total_return": metrics["total_return"],
                "val/pct_flat": metrics["pct_flat"],
                "val/pct_long": metrics["pct_long"],
                "val/pct_short": metrics["pct_short"],
                "val/best_step": val_cb.best_step,
            }
        )

        logger.info(
            f"Best val checkpoint at step {val_cb.best_step}: "
            f"Sharpe={metrics['sharpe_ratio']:.3f} | "
            f"Return={metrics['total_return']:.2%} | "
            f"MaxDD={metrics['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sweep_train()
