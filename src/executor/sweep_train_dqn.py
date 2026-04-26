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
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
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

# Step offset per seed so W&B x-axis doesn't overlap across sequential seeds
_SEED_STEP_OFFSETS = {42: 0, 123: 200_000, 999: 400_000}


class _SeedDQNWandbCallback(BaseCallback):
    """Log DQN training metrics with step offset for multi-seed runs."""

    def __init__(self, seed: int, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.seed = seed
        self.log_freq = log_freq
        self._offset = _SEED_STEP_OFFSETS.get(seed, 0)
        self._portfolio_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "portfolio_return" in info:
                self._portfolio_returns.append(info["portfolio_return"])

        if self.n_calls % self.log_freq == 0 and self._portfolio_returns:
            step = self.num_timesteps + self._offset
            recent = np.array(self._portfolio_returns[-2000:])
            metrics = {
                "train/timesteps": step,
                "train/mean_return": float(np.mean(recent)),
            }
            if len(recent) >= 50:
                from src.executor.evaluate import compute_sharpe_ratio

                metrics["train/sharpe_ratio"] = compute_sharpe_ratio(recent, annualize=True)
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        return True


def _train_single_seed_dqn(
    seed: int,
    run_dir: Path,
    cfg: dict,
    policy_kwargs: dict,
    normalizer: "StaticNormalizer",
) -> tuple[dict[str, float], dict[str, float], int]:
    """Train DQN with one seed, return (val_metrics, test_metrics, best_step)."""
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    reward_type = cfg["reward_type"]

    env_fn = make_trading_env(
        split="train",
        random_start=True,
        reward_type=reward_type,
    )
    raw_env = env_fn()
    norm_env = StaticNormWrapper(raw_env, normalizer)
    vec_env = DummyVecEnv([lambda: norm_env])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=64,
        tau=1.0,
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        target_update_interval=cfg["target_update_interval"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=cfg["exploration_final_eps"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    val_cb = DQNValCheckpointCallback(
        normalizer=normalizer,
        run_dir=seed_dir,
        eval_freq=cfg["eval_freq"],
        patience=cfg["patience"],
        min_timesteps=cfg["min_timesteps"],
        reward_type=reward_type,
    )
    train_cb = _SeedDQNWandbCallback(seed=seed, log_freq=1000)

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[val_cb, train_cb],
        progress_bar=False,
    )

    # Best val metrics
    val_metrics = val_cb.best_metrics
    if not val_metrics:
        val_metrics = evaluate_dqn_on_split(
            model=model,
            normalizer=normalizer,
            split="val",
            reward_type=reward_type,
        )

    # Reload best checkpoint for test evaluation
    best_model_path = seed_dir / "best" / "model.zip"
    if best_model_path.exists():
        best_model = DQN.load(str(best_model_path), env=vec_env)
    else:
        best_model = model

    test_metrics = evaluate_dqn_on_split(
        model=best_model,
        normalizer=normalizer,
        split="test",
        reward_type=reward_type,
    )

    vec_env.close()

    logger.info(
        f"  Seed {seed}: val Sharpe={val_metrics['sharpe_ratio']:.3f}, "
        f"test Sharpe={test_metrics['sharpe_ratio']:.3f}, "
        f"best_step={val_cb.best_step}"
    )

    return val_metrics, test_metrics, val_cb.best_step


def sweep_train() -> None:
    """Single sweep run: train 3 seeds, report mean val Sharpe."""
    SEEDS = [42, 123, 999]

    with wandb.init() as run:
        cfg = wandb.config

        params = {
            "learning_rate": cfg.get("learning_rate", 1.98e-4),
            "gamma": cfg.get("gamma", 0.97),
            "buffer_size": cfg.get("buffer_size", 50_000),
            "learning_starts": cfg.get("learning_starts", 1_000),
            "exploration_fraction": cfg.get("exploration_fraction", 0.4),
            "exploration_final_eps": cfg.get("exploration_final_eps", 0.05),
            "target_update_interval": cfg.get("target_update_interval", 1_000),
            "train_freq": cfg.get("train_freq", 4),
            "gradient_steps": cfg.get("gradient_steps", 1),
            "total_timesteps": cfg.get("total_timesteps", 100_000),
            "patience": cfg.get("patience", 25),
            "reward_type": cfg.get("reward_type", "log_return"),
            "eval_freq": 2048,
            "min_timesteps": 10_000,
        }

        # Architecture
        net_arch_width = cfg.get("net_arch_width", 64)
        net_arch_depth = cfg.get("net_arch_depth", 2)
        activation_name = cfg.get("activation_fn", "tanh")

        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU}
        activation_fn = activation_map.get(activation_name, nn.Tanh)
        net_arch = [net_arch_width] * net_arch_depth
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}

        run_dir = Path("experiments/executor_dqn/sweep") / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DQN sweep run {run.id}: lr={params['learning_rate']:.2e}, "
            f"arch={net_arch_width}x{net_arch_depth} {activation_name}, "
            f"reward={params['reward_type']}, seeds={SEEDS}"
        )

        # Load or compute normalization stats
        if DEFAULT_STATS_PATH.exists():
            normalizer = StaticNormalizer.load(DEFAULT_STATS_PATH)
        else:
            normalizer = compute_normalization_stats(reward_type=params["reward_type"])

        # Train 3 seeds
        val_sharpes = []
        test_sharpes = []
        all_val_metrics = []
        all_test_metrics = []

        for seed in SEEDS:
            val_m, test_m, best_step = _train_single_seed_dqn(
                seed=seed,
                run_dir=run_dir,
                cfg=params,
                policy_kwargs=policy_kwargs,
                normalizer=normalizer,
            )
            val_sharpes.append(val_m["sharpe_ratio"])
            test_sharpes.append(test_m["sharpe_ratio"])
            all_val_metrics.append(val_m)
            all_test_metrics.append(test_m)

        mean_val_sharpe = float(np.mean(val_sharpes))
        mean_test_sharpe = float(np.mean(test_sharpes))

        def _mean_metric(metrics_list: list[dict], key: str) -> float:
            return float(np.mean([m[key] for m in metrics_list]))

        wandb.log(
            {
                "val/sharpe_ratio": mean_val_sharpe,
                "val/total_return": _mean_metric(all_val_metrics, "total_return"),
                "val/max_drawdown": _mean_metric(all_val_metrics, "max_drawdown"),
                "val/pct_flat": _mean_metric(all_val_metrics, "pct_flat"),
                "val/pct_long": _mean_metric(all_val_metrics, "pct_long"),
                "val/pct_short": _mean_metric(all_val_metrics, "pct_short"),
                "val/sharpe_std": float(np.std(val_sharpes)),
                # Test metrics — logged for visibility, NOT optimized
                "test/sharpe_ratio": mean_test_sharpe,
                "test/total_return": _mean_metric(all_test_metrics, "total_return"),
                "test/max_drawdown": _mean_metric(all_test_metrics, "max_drawdown"),
                "test/sharpe_std": float(np.std(test_sharpes)),
                # Per-seed
                "val/sharpe_seed_42": val_sharpes[0],
                "val/sharpe_seed_123": val_sharpes[1],
                "val/sharpe_seed_999": val_sharpes[2],
            }
        )

        logger.info(
            f"Mean val Sharpe={mean_val_sharpe:.3f} (std={np.std(val_sharpes):.3f}) | "
            f"Mean test Sharpe={mean_test_sharpe:.3f} (std={np.std(test_sharpes):.3f})"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sweep_train()
