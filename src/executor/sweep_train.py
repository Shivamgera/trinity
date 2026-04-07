"""W&B sweep training script for Executor hyperparameter search.

This script is invoked by the W&B agent once per sweep run. It reads
hyperparameter values from wandb.config, trains the PPO model with
early-stopping based on validation Sharpe ratio, and logs the metric
used for model selection:
    val/sharpe_ratio

Early stopping saves the model checkpoint with the best val Sharpe
across training, then reports that as the sweep's selection metric.
This prevents overfitting — train Sharpe reaches 13+ in ~50k steps
while val Sharpe degrades with further training.

Usage (launch sweep then start agent):
    wandb sweep configs/executor_sweep.yaml
    wandb agent <sweep_id>

Or programmatically:
    python3 -m src.executor.sweep_train
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env_factory import create_vec_env, make_trading_env
from src.executor.evaluate import compute_max_drawdown, compute_sharpe_ratio
from src.executor.train import WandbCallback

logger = logging.getLogger(__name__)


def evaluate_on_split(
    model: PPO,
    vec_normalize: VecNormalize | None,
    split: str = "val",
    ticker: str = "AAPL",
    n_eval_episodes: int = 1,
) -> dict[str, float]:
    """Evaluate a trained model on a data split.

    Uses a *deterministic, sequential* walk-through of the entire split
    from the first available step to the last.  This avoids the inflated
    Sharpe / near-zero drawdown artefact caused by overlapping random-start
    episodes, which all terminate at the same final bar and therefore share
    80-90 percent of their steps.

    Args:
        model: Trained PPO model.
        vec_normalize: VecNormalize wrapper (set to eval mode first).
        split: Data split to evaluate on ("val" or "test").
        ticker: Ticker symbol.
        n_eval_episodes: How many sequential passes to run (default 1).
                         Values > 1 only make sense when episode_length is
                         set, otherwise only one full episode fits the split.

    Returns:
        Dict with sharpe_ratio, total_return, max_drawdown,
        mean_episode_reward, std_episode_reward, pct_flat/long/short.
    """
    env_fn = make_trading_env(
        ticker=ticker,
        split=split,
        random_start=False,  # deterministic: start at the beginning of the split
    )
    eval_env = DummyVecEnv([env_fn])

    if vec_normalize is not None and vec_normalize.norm_obs:
        # Wrap with frozen normalizer (don't update running stats)
        eval_vn = VecNormalize(eval_env, training=False, norm_reward=False)
        eval_vn.obs_rms = vec_normalize.obs_rms
        eval_vn.ret_rms = vec_normalize.ret_rms
        eval_vn.clip_obs = vec_normalize.clip_obs
        active_env = eval_vn
    else:
        # norm_obs=False: features are already z-normalized, no VecNormalize
        # needed for evaluation (reward normalization is irrelevant at eval)
        active_env = eval_env

    episode_rewards: list[float] = []
    portfolio_returns: list[float] = []
    actions: list[int] = []

    obs = active_env.reset()
    current_reward = 0.0

    max_steps = 10_000  # safety limit
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = active_env.step(action)
        current_reward += float(reward[0])
        actions.append(int(action[0]))

        raw_info = info[0]
        if "portfolio_return" in raw_info:
            portfolio_returns.append(raw_info["portfolio_return"])

        if done[0]:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            obs = active_env.reset()

        if len(episode_rewards) >= n_eval_episodes:
            break

    active_env.close()

    if not episode_rewards:
        episode_rewards = [0.0]

    returns_arr = np.array(portfolio_returns) if portfolio_returns else np.zeros(1)
    cum_returns = np.cumprod(1.0 + returns_arr)

    # Position distribution: action 0=flat, 1=long, 2=short
    n_actions = max(len(actions), 1)
    pct_flat = actions.count(0) / n_actions
    pct_long = actions.count(1) / n_actions
    pct_short = actions.count(2) / n_actions

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "sharpe_ratio": compute_sharpe_ratio(returns_arr, annualize=True),
        "max_drawdown": compute_max_drawdown(cum_returns),
        "total_return": float(cum_returns[-1] - 1.0),
        "pct_flat": pct_flat,
        "pct_long": pct_long,
        "pct_short": pct_short,
    }


class ValCheckpointCallback(BaseCallback):
    """Evaluate on val after each rollout; save best checkpoint.

    This implements early stopping for PPO training on small financial
    datasets where the model overfits quickly (train Sharpe > 10 after
    just 1-2 rollouts).  The best val-Sharpe checkpoint is saved and
    its metrics are stored for final reporting to the sweep agent.

    Attributes:
        best_sharpe: Best val Sharpe seen so far.
        best_metrics: Full metrics dict from the best evaluation.
        best_step: Timestep at which the best checkpoint was saved.
    """

    def __init__(
        self,
        vec_normalize: VecNormalize,
        run_dir: Path,
        patience: int = 3,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._vec_normalize = vec_normalize
        self._run_dir = run_dir
        self._patience = patience  # rollouts without improvement before stop
        self._no_improve_count = 0
        self.best_sharpe: float = -np.inf
        self.best_metrics: dict[str, float] = {}
        self.best_step: int = 0
        self._eval_count: int = 0

    def _on_rollout_end(self) -> None:
        """Called after each PPO rollout collection (before gradient update)."""
        self._eval_count += 1

        # Temporarily switch vec_normalize to eval mode
        orig_training = self._vec_normalize.training
        orig_norm_reward = self._vec_normalize.norm_reward
        self._vec_normalize.training = False
        self._vec_normalize.norm_reward = False

        metrics = evaluate_on_split(
            model=self.model,
            vec_normalize=self._vec_normalize,
            split="val",
        )

        # Restore training mode
        self._vec_normalize.training = orig_training
        self._vec_normalize.norm_reward = orig_norm_reward

        sharpe = metrics["sharpe_ratio"]
        step = self.num_timesteps

        # Log to W&B as a time-series so we can see the trajectory
        if wandb.run is not None:
            wandb.log(
                {
                    "val_rollout/sharpe_ratio": sharpe,
                    "val_rollout/total_return": metrics["total_return"],
                    "val_rollout/max_drawdown": metrics["max_drawdown"],
                    "val_rollout/pct_flat": metrics["pct_flat"],
                    "val_rollout/pct_long": metrics["pct_long"],
                    "val_rollout/pct_short": metrics["pct_short"],
                    "val_rollout/step": step,
                },
                step=step,
            )

        logger.info(
            f"  [rollout {self._eval_count}, step {step}] "
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
            self._vec_normalize.save(str(best_dir / "vec_normalize.pkl"))

            logger.info(f"  >> New best val Sharpe={sharpe:.3f} at step {step}")
        else:
            self._no_improve_count += 1
            logger.info(
                f"  >> No improvement ({self._no_improve_count}/{self._patience})"
            )

    def _on_step(self) -> bool:
        """Return False to stop training when patience is exhausted."""
        if self._no_improve_count >= self._patience:
            logger.info(
                f"Early stopping: no val improvement for {self._patience} rollouts. "
                f"Best Sharpe={self.best_sharpe:.3f} at step {self.best_step}."
            )
            return False
        return True


def sweep_train() -> None:
    """Single sweep run: train with early stopping, report best val metrics."""
    # W&B sweep agent initializes wandb automatically
    with wandb.init() as run:
        cfg = wandb.config

        # Read hyperparameters from sweep config
        learning_rate = cfg.get("learning_rate", 3e-4)
        gamma = cfg.get("gamma", 0.99)
        ent_coef = cfg.get("ent_coef", 0.01)
        gae_lambda = cfg.get("gae_lambda", 0.95)
        clip_range = cfg.get("clip_range", 0.2)
        inaction_penalty = cfg.get("inaction_penalty", 0.0)
        reward_type = cfg.get("reward_type", "log_return")
        norm_reward = cfg.get("norm_reward", False)

        # Training dynamics
        n_steps = cfg.get("n_steps", 2048)
        batch_size = cfg.get("batch_size", 64)
        n_epochs = 10                  # fixed
        n_envs = 8                     # fixed
        total_timesteps = cfg.get("total_timesteps", 1_000_000)
        patience = cfg.get("patience", 12)

        # Architecture from sweep config
        net_arch_width = cfg.get("net_arch_width", 64)
        net_arch_depth = cfg.get("net_arch_depth", 2)
        activation_name = cfg.get("activation_fn", "tanh")

        # Map activation name to torch module
        from torch import nn as _nn
        activation_map = {"tanh": _nn.Tanh, "relu": _nn.ReLU}
        activation_fn = activation_map.get(activation_name, _nn.Tanh)

        # Build net_arch: same width for all layers, separate pi/vf heads
        layer_sizes = [net_arch_width] * net_arch_depth
        net_arch = dict(pi=layer_sizes, vf=layer_sizes)

        # DSR eta only used if reward_type == "dsr"
        dsr_eta = cfg.get("dsr_eta", 0.008)

        run_dir = Path("experiments/executor/sweep") / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Sweep run {run.id}: lr={learning_rate}, n_steps={n_steps}, "
            f"gamma={gamma}, ent_coef={ent_coef}, gae_lambda={gae_lambda}, "
            f"clip_range={clip_range}, batch_size={batch_size}, "
            f"arch={net_arch_width}x{net_arch_depth} {activation_name}, "
            f"inaction_penalty={inaction_penalty}, norm_reward={norm_reward}, "
            f"reward={reward_type}, timesteps={total_timesteps}, patience={patience}"
        )

        # Build vectorized training env
        env_fns = create_vec_env(
            n_envs=n_envs,
            split="train",
            dsr_eta=dsr_eta,
            inaction_penalty=inaction_penalty,
            random_start=True,
            reward_type=reward_type,
        )
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecNormalize(
            vec_env, norm_obs=False, norm_reward=norm_reward, clip_obs=10.0
        )

        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
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
            seed=42,
            verbose=0,
        )

        # Callbacks: early-stopping val checkpoint + W&B training metrics
        val_cb = ValCheckpointCallback(
            vec_normalize=vec_env,
            run_dir=run_dir,
            patience=patience,
        )
        wandb_cb = WandbCallback(log_freq=1000)

        model.learn(
            total_timesteps=total_timesteps,
            callback=[val_cb, wandb_cb],
            progress_bar=False,
        )

        # Copy best checkpoint to run_dir root for select_best.py
        best_dir = run_dir / "best"
        if best_dir.exists():
            shutil.copy2(best_dir / "model.zip", run_dir / "model.zip")
            shutil.copy2(best_dir / "vec_normalize.pkl", run_dir / "vec_normalize.pkl")
        else:
            # No best checkpoint (shouldn't happen), save final model
            model.save(str(run_dir / "model.zip"))
            vec_env.save(str(run_dir / "vec_normalize.pkl"))

        # Report the BEST val metrics (not the final rollout's)
        metrics = val_cb.best_metrics
        if not metrics:
            # Fallback: evaluate current model if callback never ran
            vec_env.training = False
            vec_env.norm_reward = False
            metrics = evaluate_on_split(model=model, vec_normalize=vec_env, split="val")

        # Log the selection metric — W&B sweep agent uses this
        wandb.log(
            {
                "val/mean_episode_reward": metrics["mean_episode_reward"],
                "val/std_episode_reward": metrics["std_episode_reward"],
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
            f"MaxDD={metrics['max_drawdown']:.2%} | "
            f"Pos: flat={metrics['pct_flat']:.0%} "
            f"long={metrics['pct_long']:.0%} "
            f"short={metrics['pct_short']:.0%}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sweep_train()
