"""Evaluation utilities for the trained Executor PPO agent.

This module evaluates a trained model on the test split (or any split)
and generates:
    experiments/executor/evaluation/
        test_evaluation.png        — 4-panel performance overview
        return_distribution.png    — histogram of step returns
        evaluation_results.json    — numeric metrics

Usage:
    python3 -m src.executor.evaluate
    python3 -m src.executor.evaluate --model-dir experiments/executor/best_model --split test

Metrics computed:
    - Annualized Sharpe Ratio
    - Total Return
    - Maximum Drawdown
    - Position distribution (% flat / long / short)
    - Episode reward statistics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.executor.env_factory import make_trading_env
from src.executor.policy import load_executor
from src.utils.data import load_raw_ohlcv, load_numeric_features

logger = logging.getLogger(__name__)

EVAL_DIR = Path("experiments/executor/evaluation")
DEFAULT_MODEL_DIR = Path("experiments/executor/best_model")


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------


def compute_sharpe_ratio(
    returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Compute the Sharpe Ratio from a sequence of returns.

    Args:
        returns: Array of per-step portfolio returns.
        annualize: If True, scale by sqrt(trading_days).
        trading_days: Number of trading days per year (used for annualization).

    Returns:
        Sharpe ratio (0.0 if std is 0 or returns is empty).
    """
    if len(returns) == 0:
        return 0.0
    std = np.std(returns)
    if std < 1e-12:
        return 0.0
    mean = np.mean(returns)
    sharpe = mean / std
    if annualize:
        sharpe = sharpe * np.sqrt(trading_days)
    return float(sharpe)


def compute_sortino_ratio(
    returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Compute the Sortino Ratio (penalises only downside volatility).

    Uses a target return of 0 (standard convention).  Downside deviation
    is ``sqrt(mean(min(r, 0)^2))``.

    Args:
        returns: Array of per-step portfolio returns.
        annualize: If True, scale by sqrt(trading_days).
        trading_days: Number of trading days per year.

    Returns:
        Sortino ratio (0.0 if downside deviation is 0 or returns is empty).
    """
    if len(returns) == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return 0.0
    sortino = np.mean(returns) / downside_std
    if annualize:
        sortino = sortino * np.sqrt(trading_days)
    return float(sortino)


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Compute maximum drawdown from a cumulative return series.

    Args:
        cumulative_returns: Array of cumulative (compound) return values,
                            where 1.0 represents the starting value.

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 → -15% peak-to-trough).
        Returns 0.0 if the series is empty.
    """
    if len(cumulative_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.where(peak == 0, 1.0, peak)
    return float(-np.min(drawdown))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_executor(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    split: str = "test",
    ticker: str = "AAPL",
    n_eval_episodes: int = 1,
    use_wandb: bool = False,
) -> dict:
    """Evaluate a trained Executor on a data split and generate plots.

    Runs a *single deterministic sequential pass* over the full split.
    Do not set n_eval_episodes > 1 with a deterministic policy and
    random_start=False: you would just repeat the identical trajectory
    multiple times, inflating the apparent sample size without adding
    any new information.

    To benchmark against a naive baseline, compare val/test Sharpe against
    a buy-and-hold policy on the same period.  AAPL H1-2024 buy-and-hold
    Sharpe is ~2.0+, so a model Sharpe near that level is not evidence of
    skill — it may simply be long-bias in a bull-trend window.

    Args:
        model_dir: Directory containing ``model.zip`` and optionally
                   ``vec_normalize.pkl``.
        split: Data split to evaluate on ("test", "val", etc.).
        ticker: Ticker symbol.
        n_eval_episodes: Number of sequential passes (default 1).
        use_wandb: If True, log metrics to W&B.

    Returns:
        Dict of evaluation metrics.
    """
    model, vec_normalize = load_executor(model_dir)

    # Build evaluation environment (no random start for deterministic eval)
    env_fn = make_trading_env(
        ticker=ticker,
        split=split,
        random_start=False,
    )
    eval_env = DummyVecEnv([env_fn])

    if vec_normalize is not None:
        eval_vn = VecNormalize(eval_env, training=False, norm_reward=False)
        eval_vn.obs_rms = vec_normalize.obs_rms
        eval_vn.ret_rms = vec_normalize.ret_rms
        eval_vn.clip_obs = vec_normalize.clip_obs
        active_env = eval_vn
    else:
        active_env = eval_env

    # Rollout collection
    all_portfolio_returns: list[float] = []
    all_positions: list[float] = []
    all_rewards: list[float] = []
    episode_rewards: list[float] = []

    obs = active_env.reset()
    current_ep_reward = 0.0
    episodes = 0

    max_steps = 50_000  # safety limit
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = active_env.step(action)

        current_ep_reward += float(reward[0])
        all_rewards.append(float(reward[0]))

        raw_info = info[0]
        if "portfolio_return" in raw_info:
            all_portfolio_returns.append(raw_info["portfolio_return"])
        if "position" in raw_info:
            all_positions.append(raw_info["position"])

        if done[0]:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            episodes += 1
            obs = active_env.reset()

        if episodes >= n_eval_episodes:
            break

    active_env.close()

    # --------------------------------------------------------- compute metrics
    returns_arr = np.array(all_portfolio_returns) if all_portfolio_returns else np.zeros(1)
    positions_arr = np.array(all_positions) if all_positions else np.zeros(1)

    cumulative_returns = np.cumprod(1.0 + returns_arr)
    total_return = float(cumulative_returns[-1] - 1.0) if len(cumulative_returns) else 0.0
    sharpe = compute_sharpe_ratio(returns_arr, annualize=True)
    max_dd = compute_max_drawdown(cumulative_returns)

    # --------------------------------------------------------- buy-and-hold benchmark
    # Load raw close prices aligned to the same steps the agent actually traded.
    # The agent starts at `lookback_window` days into the split (deterministic
    # eval with random_start=False), so we offset the B&H series to match.
    # This is the only honest baseline: same dates, no transaction costs
    # (buy-and-hold pays cost once at entry/exit, effectively zero per-step).
    raw_ohlcv = load_raw_ohlcv(ticker=ticker)
    features_df = load_numeric_features(ticker=ticker, split=split)
    bah_prices = raw_ohlcv["Close"].reindex(features_df.index).ffill()
    # Agent starts at lookback_window (default 10) into the split, so
    # offset the B&H prices to the same starting date.
    lookback_window = 10  # must match env default
    bah_prices = bah_prices.iloc[lookback_window:]
    bah_prices = bah_prices.iloc[:len(returns_arr) + 1]  # trim to actual steps taken
    if len(bah_prices) >= 2:
        bah_returns = bah_prices.pct_change().dropna().values
        # Trim to exact length as agent trajectory
        bah_returns = bah_returns[:len(returns_arr)]
        bah_cum = np.cumprod(1.0 + bah_returns)
        bah_sharpe = compute_sharpe_ratio(bah_returns, annualize=True)
        bah_total_return = float(bah_cum[-1] - 1.0)
        bah_max_dd = compute_max_drawdown(bah_cum)
    else:
        bah_returns = np.zeros(1)
        bah_cum = np.ones(1)
        bah_sharpe = 0.0
        bah_total_return = 0.0
        bah_max_dd = 0.0

    pos_counts = {
        "pct_flat": float(np.mean(positions_arr == 0.0)),
        "pct_long": float(np.mean(positions_arr == 1.0)),
        "pct_short": float(np.mean(positions_arr == -1.0)),
    }

    metrics = {
        "split": split,
        "n_steps": len(returns_arr),
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_episode_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        # Buy-and-hold benchmark (same period, no skill required)
        "bah_total_return": bah_total_return,
        "bah_sharpe_ratio": bah_sharpe,
        "bah_max_drawdown": bah_max_dd,
        # Alpha vs benchmark
        "alpha_return": total_return - bah_total_return,
        "alpha_sharpe": sharpe - bah_sharpe,
        **pos_counts,
    }

    logger.info(
        f"\nEvaluation results ({split}):\n"
        f"  {'Metric':<22} {'Agent':>10}  {'Buy&Hold':>10}  {'Alpha':>10}\n"
        f"  {'-'*54}\n"
        f"  {'Total Return':<22} {total_return:>10.2%}  {bah_total_return:>10.2%}  {total_return - bah_total_return:>+10.2%}\n"
        f"  {'Sharpe Ratio':<22} {sharpe:>10.3f}  {bah_sharpe:>10.3f}  {sharpe - bah_sharpe:>+10.3f}\n"
        f"  {'Max Drawdown':<22} {max_dd:>10.2%}  {bah_max_dd:>10.2%}  {max_dd - bah_max_dd:>+10.2%}\n"
        f"  Position mix: long={pos_counts['pct_long']:.1%}, "
        f"flat={pos_counts['pct_flat']:.1%}, short={pos_counts['pct_short']:.1%}"
    )

    # --------------------------------------------------------------- save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = EVAL_DIR / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------------------- plots
    _plot_evaluation(
        cumulative_returns=cumulative_returns,
        bah_cumulative_returns=bah_cum,
        positions=positions_arr,
        rewards=np.array(all_rewards),
        portfolio_returns=returns_arr,
        metrics=metrics,
        output_dir=EVAL_DIR,
    )

    if use_wandb:
        from src.utils.logging import log_metrics
        log_metrics({f"{split}/{k}": v for k, v in metrics.items()}, step=0)

    return metrics


def _plot_evaluation(
    cumulative_returns: np.ndarray,
    bah_cumulative_returns: np.ndarray,
    positions: np.ndarray,
    rewards: np.ndarray,
    portfolio_returns: np.ndarray,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Generate and save evaluation plots.

    Creates two files:
        {output_dir}/test_evaluation.png  — 4-panel overview with buy-and-hold
        {output_dir}/return_distribution.png — return histogram
    """
    # ---- 4-panel evaluation overview ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Executor Evaluation — {metrics['split'].title()} Split\n"
        f"Agent  Sharpe: {metrics['sharpe_ratio']:.3f} | Return: {metrics['total_return']:.2%} | MaxDD: {metrics['max_drawdown']:.2%}\n"
        f"B&Hold Sharpe: {metrics['bah_sharpe_ratio']:.3f} | Return: {metrics['bah_total_return']:.2%} | MaxDD: {metrics['bah_max_drawdown']:.2%}  "
        f"(α={metrics['alpha_sharpe']:+.3f})",
        fontsize=11,
        fontweight="bold",
    )

    # Panel 1: Cumulative returns vs buy-and-hold
    ax1 = axes[0, 0]
    ax1.plot(cumulative_returns, color="steelblue", linewidth=1.2, label="Agent")
    # Align buy-and-hold to same length
    bah_plot = bah_cumulative_returns[:len(cumulative_returns)]
    ax1.plot(bah_plot, color="darkorange", linewidth=1.0, linestyle="--", label="Buy & Hold")
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_title("Cumulative Return vs Buy & Hold")
    ax1.set_ylabel("Portfolio Value")
    ax1.set_xlabel("Step")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    ax2 = axes[0, 1]
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown_series = (cumulative_returns - peak) / np.where(peak == 0, 1.0, peak)
    ax2.fill_between(range(len(drawdown_series)), drawdown_series, 0, color="red", alpha=0.4)
    ax2.set_title(f"Drawdown (max: -{metrics['max_drawdown']:.2%})")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Position over time
    ax3 = axes[1, 0]
    cmap = {-1.0: "red", 0.0: "gray", 1.0: "green"}
    colors = [cmap.get(p, "gray") for p in positions[:1000]]  # first 1000 for readability
    ax3.scatter(range(len(colors)), positions[:1000], c=colors, s=2, alpha=0.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"])
    ax3.set_title("Position Over Time (first 1000 steps)")
    ax3.set_xlabel("Step")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Position distribution
    ax4 = axes[1, 1]
    pos_labels = ["Short (-1)", "Flat (0)", "Long (1)"]
    pos_values = [
        metrics["pct_short"],
        metrics["pct_flat"],
        metrics["pct_long"],
    ]
    bar_colors = ["red", "gray", "green"]
    bars = ax4.bar(pos_labels, pos_values, color=bar_colors, alpha=0.75)
    for bar, v in zip(bars, pos_values):
        ax4.text(bar.get_x() + bar.get_width() / 2.0, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax4.set_title("Position Distribution")
    ax4.set_ylabel("Fraction of Steps")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "test_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Return distribution histogram ----
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(portfolio_returns, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(float(np.mean(portfolio_returns)), color="green", linestyle="-", linewidth=1.2, label=f"Mean: {np.mean(portfolio_returns):.4%}")
    ax.set_title("Distribution of Per-Step Portfolio Returns")
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "return_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    logger.info(f"Plots saved to {output_dir}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate trained Executor agent")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory with model.zip (default: experiments/executor/best_model)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to W&B")
    args = parser.parse_args()

    evaluate_executor(
        model_dir=args.model_dir,
        split=args.split,
        ticker=args.ticker,
        n_eval_episodes=args.n_episodes,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
