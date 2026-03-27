"""DQN training script for the Executor agent.

Parallel pipeline to PPO — shares the same TradingEnv but uses:
- DQN (off-policy, value-based) instead of PPO (on-policy, actor-critic)
- StaticNormWrapper instead of VecNormalize (no replay buffer distribution shift)
- DQNValCheckpointCallback with _on_step frequency check instead of _on_rollout_end

Usage:
    python3 -m src.executor.train_dqn
    python3 -m src.executor.train_dqn --timesteps 250000 --seed 42
    python3 -m src.executor.train_dqn --timesteps 10000 --no-wandb  # smoke test

The trained model is saved to:
    experiments/executor_dqn/run_<timestamp>/model.zip
    experiments/executor_dqn/run_<timestamp>/normalization_stats.json
    experiments/executor_dqn/run_<timestamp>/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from src.executor.env_factory import make_trading_env
from src.executor.evaluate import compute_max_drawdown, compute_sharpe_ratio
from src.executor.static_normalize import (
    StaticNormalizer,
    StaticNormWrapper,
    compute_normalization_stats,
    DEFAULT_STATS_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helper (DQN-specific: uses StaticNormalizer, not VecNormalize)
# ---------------------------------------------------------------------------

def evaluate_dqn_on_split(
    model: DQN,
    normalizer: StaticNormalizer | None,
    split: str = "val",
    ticker: str = "AAPL",
    reward_type: str = "log_return",
) -> dict[str, float]:
    """Evaluate a DQN model on a data split using deterministic sequential rollout.

    Mirrors sweep_train.evaluate_on_split() but uses StaticNormWrapper
    instead of VecNormalize.

    Args:
        model: Trained DQN model.
        normalizer: StaticNormalizer (or None to skip normalization).
        split: Data split ("val" or "test").
        ticker: Ticker symbol.
        reward_type: Reward function type.

    Returns:
        Dict with sharpe_ratio, total_return, max_drawdown,
        mean_episode_reward, pct_flat/long/short.
    """
    env_fn = make_trading_env(
        ticker=ticker,
        split=split,
        random_start=False,
        reward_type=reward_type,
    )
    raw_env = env_fn()

    # Wrap with static normalizer if available
    if normalizer is not None:
        eval_env = StaticNormWrapper(raw_env, normalizer)
    else:
        eval_env = raw_env

    # Wrap in DummyVecEnv for model.predict() compatibility
    eval_vec = DummyVecEnv([lambda: eval_env])

    portfolio_returns: list[float] = []
    actions: list[int] = []
    episode_rewards: list[float] = []
    current_reward = 0.0

    obs = eval_vec.reset()
    max_steps = 10_000
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_vec.step(action)
        current_reward += float(reward[0])
        actions.append(int(action[0]))

        raw_info = info[0]
        if "portfolio_return" in raw_info:
            portfolio_returns.append(raw_info["portfolio_return"])

        if done[0]:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            break  # Single deterministic episode

    eval_vec.close()

    if not episode_rewards:
        episode_rewards = [0.0]

    returns_arr = np.array(portfolio_returns) if portfolio_returns else np.zeros(1)
    cum_returns = np.cumprod(1.0 + returns_arr)

    n_actions = max(len(actions), 1)
    pct_flat = actions.count(0) / n_actions
    pct_long = actions.count(1) / n_actions
    pct_short = actions.count(2) / n_actions

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)) if len(episode_rewards) > 1 else 0.0,
        "sharpe_ratio": compute_sharpe_ratio(returns_arr, annualize=True),
        "max_drawdown": compute_max_drawdown(cum_returns),
        "total_return": float(cum_returns[-1] - 1.0),
        "pct_flat": pct_flat,
        "pct_long": pct_long,
        "pct_short": pct_short,
    }


# ---------------------------------------------------------------------------
# DQN-specific early-stopping callback
# ---------------------------------------------------------------------------

class DQNValCheckpointCallback(BaseCallback):
    """Evaluate DQN on val split periodically; save best checkpoint.

    DQN is off-policy: _on_rollout_end never fires (no rollout collection).
    Instead, we use _on_step with a frequency counter to trigger evaluation.

    Attributes:
        best_sharpe: Best val Sharpe seen so far.
        best_metrics: Full metrics dict from the best evaluation.
        best_step: Timestep at which the best checkpoint was saved.
    """

    def __init__(
        self,
        normalizer: StaticNormalizer | None,
        run_dir: Path,
        eval_freq: int = 2048,
        patience: int = 10,
        reward_type: str = "log_return",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._normalizer = normalizer
        self._run_dir = run_dir
        self._eval_freq = eval_freq
        self._patience = patience
        self._reward_type = reward_type
        self._no_improve_count = 0
        self.best_sharpe: float = -np.inf
        self.best_metrics: dict[str, float] = {}
        self.best_step: int = 0
        self._eval_count: int = 0

    def _on_step(self) -> bool:
        """Called every env step. Evaluate when step is multiple of eval_freq."""
        if self.num_timesteps % self._eval_freq != 0:
            return True

        self._eval_count += 1

        metrics = evaluate_dqn_on_split(
            model=self.model,
            normalizer=self._normalizer,
            split="val",
            reward_type=self._reward_type,
        )

        sharpe = metrics["sharpe_ratio"]
        step = self.num_timesteps

        # Log to W&B if active
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "val_step/sharpe_ratio": sharpe,
                        "val_step/total_return": metrics["total_return"],
                        "val_step/max_drawdown": metrics["max_drawdown"],
                        "val_step/pct_flat": metrics["pct_flat"],
                        "val_step/pct_long": metrics["pct_long"],
                        "val_step/pct_short": metrics["pct_short"],
                        "val_step/step": step,
                    },
                    step=step,
                )
        except ImportError:
            pass

        logger.info(
            f"  [eval {self._eval_count}, step {step}] "
            f"val Sharpe={sharpe:.3f} ret={metrics['total_return']:.2%} "
            f"mdd={metrics['max_drawdown']:.2%} "
            f"pos=F{metrics['pct_flat']:.0%}/L{metrics['pct_long']:.0%}/S{metrics['pct_short']:.0%}"
        )

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_metrics = metrics
            self.best_step = step
            self._no_improve_count = 0

            # Save best checkpoint
            best_dir = self._run_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(best_dir / "model.zip"))

            # Copy normalization stats alongside model
            if self._normalizer is not None:
                self._normalizer.save(best_dir / "normalization_stats.json")

            logger.info(f"  >> New best val Sharpe={sharpe:.3f} at step {step}")
        else:
            self._no_improve_count += 1
            logger.info(
                f"  >> No improvement ({self._no_improve_count}/{self._patience})"
            )

        # Early stopping
        if self._no_improve_count >= self._patience:
            logger.info(
                f"Early stopping: no val improvement for {self._patience} eval periods. "
                f"Best Sharpe={self.best_sharpe:.3f} at step {self.best_step}."
            )
            return False

        return True


# ---------------------------------------------------------------------------
# DQN W&B training callback
# ---------------------------------------------------------------------------

class DQNWandbCallback(BaseCallback):
    """Log DQN training metrics to W&B at regular step intervals.

    DQN doesn't have rollout-end events, so we log at fixed step intervals.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self._portfolio_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "portfolio_return" in info:
                self._portfolio_returns.append(info["portfolio_return"])

        if self.n_calls % self.log_freq == 0 and self._portfolio_returns:
            try:
                import wandb
                if wandb.run is not None:
                    recent = np.array(self._portfolio_returns[-2000:])
                    metrics = {
                        "train/timesteps": self.num_timesteps,
                        "train/mean_return": float(np.mean(recent)),
                    }
                    if len(recent) >= 50:
                        metrics["train/sharpe_ratio"] = compute_sharpe_ratio(
                            recent, annualize=True
                        )
                    wandb.log(metrics, step=self.num_timesteps)
            except ImportError:
                pass

        return True


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_dqn(
    ticker: str = "AAPL",
    total_timesteps: int = 250_000,
    learning_rate: float = 1.98e-4,
    gamma: float = 0.95,
    buffer_size: int = 50_000,
    learning_starts: int = 1_000,
    exploration_fraction: float = 0.4,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1_000,
    train_freq: int = 4,
    gradient_steps: int = 1,
    batch_size: int = 64,
    tau: float = 1.0,
    net_arch: list[int] | None = None,
    eval_freq: int = 2048,
    patience: int = 10,
    reward_type: str = "log_return",
    seed: int = 42,
    use_wandb: bool = True,
    run_dir: Path | None = None,
    stats_path: str | Path | None = None,
) -> tuple[DQN, StaticNormalizer | None, Path]:
    """Train a DQN agent on the TradingEnv with static normalization.

    Args:
        ticker: Ticker symbol.
        total_timesteps: Max training steps (early stopping usually fires sooner).
        learning_rate: DQN learning rate.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        learning_starts: Steps of random exploration before learning begins.
        exploration_fraction: Fraction of total_timesteps for epsilon decay.
        exploration_initial_eps: Starting epsilon (exploration rate).
        exploration_final_eps: Final epsilon after decay.
        target_update_interval: Steps between target network hard updates.
        train_freq: Train every N env steps.
        gradient_steps: Gradient steps per train_freq trigger.
        batch_size: Mini-batch size for replay sampling.
        tau: Target network soft update coefficient (1.0 = hard update).
        net_arch: Q-network hidden layer sizes (default [64, 64]).
        eval_freq: Steps between val evaluations.
        patience: Eval periods without val improvement before early stopping.
        reward_type: Reward function.
        seed: Random seed.
        use_wandb: Log to W&B.
        run_dir: Output directory. Auto-generated if None.
        stats_path: Path to pre-computed normalization stats JSON.
                    If None, computes fresh stats.

    Returns:
        Tuple of (trained DQN model, StaticNormalizer, output directory).
    """
    if net_arch is None:
        net_arch = [64, 64]

    # ------------------------------------------------------------------ output
    if run_dir is None:
        timestamp = int(time.time())
        run_dir = Path("experiments/executor_dqn") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": "DQN",
        "ticker": ticker,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "tau": tau,
        "net_arch": net_arch,
        "eval_freq": eval_freq,
        "patience": patience,
        "reward_type": reward_type,
        "seed": seed,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----------------------------------------------------------------- logging
    if use_wandb:
        from src.utils.logging import init_wandb
        init_wandb(phase="dqn", component="executor", config=config)
        logger.info("W&B run initialized")

    # -------------------------------------------------------- static normalizer
    if stats_path is not None and Path(stats_path).exists():
        normalizer = StaticNormalizer.load(stats_path)
        logger.info(f"Loaded normalization stats from {stats_path}")
    elif DEFAULT_STATS_PATH.exists():
        normalizer = StaticNormalizer.load(DEFAULT_STATS_PATH)
        logger.info(f"Loaded normalization stats from {DEFAULT_STATS_PATH}")
    else:
        logger.info("No pre-computed normalization stats found. Computing...")
        normalizer = compute_normalization_stats(
            ticker=ticker,
            reward_type=reward_type,
        )

    # Save a copy in run_dir
    normalizer.save(run_dir / "normalization_stats.json")

    # ------------------------------------------------------------- environment
    # DQN requires a single env (for replay buffer).
    # Wrap with StaticNormWrapper instead of VecNormalize.
    logger.info("Creating training environment with static normalization...")

    env_fn = make_trading_env(
        ticker=ticker,
        split="train",
        random_start=True,
        reward_type=reward_type,
    )
    raw_env = env_fn()
    norm_env = StaticNormWrapper(raw_env, normalizer)
    vec_env = DummyVecEnv([lambda: norm_env])

    # --------------------------------------------------------------- DQN model
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
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"DQN model has {param_count} parameters")
    logger.info(f"Training for up to {total_timesteps:,} timesteps (patience={patience})...")

    # ---------------------------------------------------------------- callbacks
    val_cb = DQNValCheckpointCallback(
        normalizer=normalizer,
        run_dir=run_dir,
        eval_freq=eval_freq,
        patience=patience,
        reward_type=reward_type,
    )

    callbacks = [val_cb]
    if use_wandb:
        callbacks.append(DQNWandbCallback(log_freq=1000))

    # ---------------------------------------------------------------- training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ---------------------------------------------------------- save final/best
    best_dir = run_dir / "best"
    if best_dir.exists() and (best_dir / "model.zip").exists():
        # Copy best to run_dir root
        shutil.copy2(best_dir / "model.zip", run_dir / "model.zip")
        if (best_dir / "normalization_stats.json").exists():
            shutil.copy2(
                best_dir / "normalization_stats.json",
                run_dir / "normalization_stats.json",
            )
    else:
        # No best checkpoint saved (shouldn't happen), save final model
        model.save(str(run_dir / "model.zip"))

    logger.info(f"Best val Sharpe={val_cb.best_sharpe:.3f} at step {val_cb.best_step}")
    logger.info(f"Model saved to {run_dir}")

    if use_wandb:
        from src.utils.logging import finish_wandb
        finish_wandb()

    return model, normalizer, run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train DQN Executor agent")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=1.98e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    args = parser.parse_args()

    train_dqn(
        ticker=args.ticker,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        patience=args.patience,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        run_dir=args.run_dir,
        stats_path=args.stats_path,
    )


if __name__ == "__main__":
    main()
