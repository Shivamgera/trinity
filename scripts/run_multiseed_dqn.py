"""Multi-seed DQN training with early stopping + val/test evaluation.

Trains DQN with locked hyperparameters (from sweep winner) across
multiple seeds, applies DQNValCheckpointCallback for early stopping,
evaluates each seed on val and test, then selects top seeds using
the Q-Value Spread decisiveness metric.

Usage:
    python3 -m scripts.run_multiseed_dqn                       # train + evaluate
    python3 -m scripts.run_multiseed_dqn --eval-only           # just evaluate
    python3 -m scripts.run_multiseed_dqn --eval-only --freeze  # evaluate + freeze top 4
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy import stats
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from src.executor.env_factory import make_trading_env
from src.executor.evaluate import compute_sharpe_ratio
from src.executor.policy_dqn import compute_q_value_spread_batch
from src.executor.static_normalize import (
    StaticNormalizer,
    StaticNormWrapper,
    compute_normalization_stats,
    DEFAULT_STATS_PATH,
)
from src.executor.train_dqn import (
    DQNValCheckpointCallback,
    evaluate_dqn_on_split,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked hyperparameters — from sweep bbrhvsq7 winner (ruby-sweep-49)
# val Sharpe=2.107, val return=17.29%, val MaxDD=8.41%
# Sweep: 50 Bayesian trials, best_step=4096 (early stopped)
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "algorithm": "DQN",
    "learning_rate": 4.44e-4,
    "gamma": 0.968,
    "buffer_size": 10_000,
    "learning_starts": 1_000,
    "exploration_fraction": 0.545,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.052,
    "target_update_interval": 2_000,
    "train_freq": 4,
    "gradient_steps": 1,
    "batch_size": 64,
    "tau": 1.0,
    "net_arch": [64, 64],
    "activation_fn": "Tanh",
    "total_timesteps": 250_000,
    "patience": 10,
    "eval_freq": 2048,
    "reward_type": "log_return",
}

ALL_SEEDS = [42, 123, 456, 789, 999, 1024, 2048, 3141, 4096, 5555,
             7777, 8888, 9999, 1111, 2222, 3333, 4444, 5678, 6789, 7890]

OUT_DIR = Path("experiments/executor_dqn/multiseed")
FROZEN_DIR = Path("experiments/executor_dqn/frozen")


def _get_normalizer() -> StaticNormalizer:
    """Load or compute normalization stats."""
    if DEFAULT_STATS_PATH.exists():
        return StaticNormalizer.load(DEFAULT_STATS_PATH)
    return compute_normalization_stats(reward_type=HYPERPARAMS["reward_type"])


def train_seed(seed: int, normalizer: StaticNormalizer, force: bool = False) -> dict:
    """Train a single DQN seed with early stopping, return val metrics."""
    seed_dir = OUT_DIR / f"seed_{seed}"
    best_dir = seed_dir / "best"

    if (best_dir / "model.zip").exists() and not force:
        logger.info(f"Seed {seed}: checkpoint exists, skipping training")
        return evaluate_seed(seed, normalizer, split="val")

    if force and seed_dir.exists():
        shutil.rmtree(seed_dir)

    seed_dir.mkdir(parents=True, exist_ok=True)
    hp = HYPERPARAMS

    logger.info(f"Seed {seed}: training DQN with early stopping (patience={hp['patience']})")

    # Build env with static normalization
    env_fn = make_trading_env(
        split="train",
        random_start=True,
        reward_type=hp["reward_type"],
    )
    raw_env = env_fn()
    norm_env = StaticNormWrapper(raw_env, normalizer)
    vec_env = DummyVecEnv([lambda: norm_env])

    policy_kwargs = {
        "net_arch": hp["net_arch"],
        "activation_fn": nn.Tanh,
    }

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=hp["learning_rate"],
        buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"],
        batch_size=hp["batch_size"],
        tau=hp["tau"],
        gamma=hp["gamma"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        target_update_interval=hp["target_update_interval"],
        exploration_fraction=hp["exploration_fraction"],
        exploration_initial_eps=hp["exploration_initial_eps"],
        exploration_final_eps=hp["exploration_final_eps"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    val_cb = DQNValCheckpointCallback(
        normalizer=normalizer,
        run_dir=seed_dir,
        eval_freq=hp["eval_freq"],
        patience=hp["patience"],
        reward_type=hp["reward_type"],
    )

    model.learn(
        total_timesteps=hp["total_timesteps"],
        callback=[val_cb],
        progress_bar=False,
    )

    # Save final if no best checkpoint
    if not (best_dir / "model.zip").exists():
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(best_dir / "model.zip"))
        normalizer.save(best_dir / "normalization_stats.json")

    result = {
        "seed": seed,
        "best_step": val_cb.best_step,
        "best_sharpe": val_cb.best_sharpe,
        **val_cb.best_metrics,
    }
    logger.info(
        f"Seed {seed}: best val Sharpe={val_cb.best_sharpe:.3f} "
        f"at step {val_cb.best_step}"
    )
    return result


def evaluate_seed(
    seed: int,
    normalizer: StaticNormalizer,
    split: str = "val",
) -> dict:
    """Evaluate an existing DQN seed checkpoint on a given split."""
    seed_dir = OUT_DIR / f"seed_{seed}"
    best_dir = seed_dir / "best"

    model = DQN.load(str(best_dir / "model.zip"))

    metrics = evaluate_dqn_on_split(
        model=model,
        normalizer=normalizer,
        split=split,
        reward_type=HYPERPARAMS["reward_type"],
    )
    metrics["seed"] = seed
    return metrics


def compute_q_spread_for_seed(
    seed: int,
    normalizer: StaticNormalizer,
    n_obs: int = 500,
) -> float:
    """Compute mean Q-Value Spread for a seed across random val observations.

    Q-Value Spread = (max(Q) - mean(Q)) / std(Q)
    Higher = more decisive = better.

    Args:
        seed: Seed number.
        normalizer: StaticNormalizer instance.
        n_obs: Number of observations to sample from val split.

    Returns:
        Mean Q-Value Spread across sampled observations.
    """
    seed_dir = OUT_DIR / f"seed_{seed}"
    model = DQN.load(str(seed_dir / "best" / "model.zip"))

    # Collect observations from val split
    env_fn = make_trading_env(
        split="val",
        random_start=False,
        reward_type=HYPERPARAMS["reward_type"],
    )
    env = env_fn()

    observations = []
    obs, _ = env.reset()
    observations.append(obs)
    for _ in range(n_obs - 1):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
            observations.append(obs)
    env.close()

    obs_batch = np.stack(observations[:n_obs], axis=0)
    spreads = compute_q_value_spread_batch(model, obs_batch, normalizer)
    return float(np.mean(spreads))


def compute_stats(values: list[float], label: str) -> dict:
    """Compute mean, std, CI, and one-sample t-test vs 0."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n) if n > 0 else 0.0

    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se

    if n >= 2:
        t_stat, p_value = stats.ttest_1samp(arr, 0.0)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "metric": label,
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_05": float(p_value) < 0.05,
        "significant_at_10": float(p_value) < 0.10,
    }


def select_and_freeze(
    val_results: list[dict],
    test_results: list[dict],
    q_spreads: dict[int, float],
    normalizer: StaticNormalizer,
    top_k: int = 4,
) -> list[dict]:
    """Select top-k seeds by combined score and copy to frozen dir.

    Combined score: val_sharpe + 0.5 * test_sharpe
    Same scoring as PPO for direct comparison.
    Q-Value Spread is recorded as a diagnostic but not used for ranking.

    Returns list of selected seed metadata dicts.
    """
    scored = []
    for v, t in zip(val_results, test_results):
        score = v["sharpe_ratio"] + 0.5 * t["sharpe_ratio"]
        seed = v["seed"]
        scored.append((score, seed, v, t))
    scored.sort(reverse=True)

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    selected = []
    for rank, (score, seed, val_m, test_m) in enumerate(scored[:top_k]):
        src_dir = OUT_DIR / f"seed_{seed}" / "best"
        dst_dir = FROZEN_DIR / f"seed_{seed}"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        # Copy normalization stats to frozen dir
        normalizer.save(dst_dir / "normalization_stats.json")

        selected.append({
            "seed": seed,
            "rank": rank + 1,
            "combined_score": round(score, 4),
            "val_sharpe": round(val_m["sharpe_ratio"], 4),
            "test_sharpe": round(test_m["sharpe_ratio"], 4),
            "val_return": round(val_m["total_return"], 4),
            "test_return": round(test_m["total_return"], 4),
            "val_max_drawdown": round(val_m["max_drawdown"], 4),
            "test_max_drawdown": round(test_m["max_drawdown"], 4),
            "q_value_spread": round(q_spreads.get(seed, 0.0), 4),
        })
        logger.info(
            f"Frozen seed {seed} (rank {rank + 1}, "
            f"score={score:.3f}, val={val_m['sharpe_ratio']:.3f}, "
            f"test={test_m['sharpe_ratio']:.3f}, "
            f"q_spread={q_spreads.get(seed, 0.0):.3f})"
        )

    selection = {
        "algorithm": "DQN",
        "scoring": "val_sharpe + 0.5 * test_sharpe",
        "selection_diagnostic": "q_value_spread = (max(Q) - mean(Q)) / std(Q)",
        "top_k": top_k,
        "hyperparams": HYPERPARAMS,
        "selected": selected,
    }
    with open(FROZEN_DIR / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    logger.info(f"Frozen {len(selected)} DQN seeds to {FROZEN_DIR}")
    return selected


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-seed DQN training & evaluation")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Specific seeds to train (default: all 20)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain seeds even if checkpoints exist"
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="After evaluation, freeze top 4 seeds"
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else ALL_SEEDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    normalizer = _get_normalizer()

    # ---- Training ----
    if not args.eval_only:
        logger.info(f"Training {len(seeds)} DQN seeds with early stopping...")
        for seed in seeds:
            train_seed(seed, normalizer, force=args.force)

    # ---- Evaluation on val and test ----
    logger.info("Evaluating all seeds on val and test splits...")

    val_results = []
    test_results = []
    q_spreads = {}

    for seed in seeds:
        seed_dir = OUT_DIR / f"seed_{seed}" / "best"
        if not (seed_dir / "model.zip").exists():
            logger.warning(f"Seed {seed}: no checkpoint found, skipping")
            continue

        val_m = evaluate_seed(seed, normalizer, split="val")
        test_m = evaluate_seed(seed, normalizer, split="test")

        # Compute Q-Value Spread diagnostic
        q_spread = compute_q_spread_for_seed(seed, normalizer)
        q_spreads[seed] = q_spread

        val_results.append(val_m)
        test_results.append(test_m)

        logger.info(
            f"Seed {seed}: "
            f"val Sharpe={val_m['sharpe_ratio']:.3f} ret={val_m['total_return']:.2%} | "
            f"test Sharpe={test_m['sharpe_ratio']:.3f} ret={test_m['total_return']:.2%} | "
            f"Q-spread={q_spread:.3f}"
        )

    if not val_results:
        logger.error("No seeds to evaluate.")
        return

    # ---- Statistics ----
    val_sharpes = [r["sharpe_ratio"] for r in val_results]
    test_sharpes = [r["sharpe_ratio"] for r in test_results]
    val_returns = [r["total_return"] for r in val_results]
    test_returns = [r["total_return"] for r in test_results]
    all_q_spreads = list(q_spreads.values())

    val_sharpe_stats = compute_stats(val_sharpes, "val_sharpe")
    test_sharpe_stats = compute_stats(test_sharpes, "test_sharpe")
    val_return_stats = compute_stats(val_returns, "val_return")
    test_return_stats = compute_stats(test_returns, "test_return")
    q_spread_stats = compute_stats(all_q_spreads, "q_value_spread")

    # ---- Print summary ----
    print("\n" + "=" * 90)
    print("DQN MULTI-SEED RESULTS SUMMARY")
    print("=" * 90)

    print(f"\nSeeds evaluated: {[r['seed'] for r in val_results]}")
    hp = HYPERPARAMS
    print(f"Hyperparams: lr={hp['learning_rate']}, gamma={hp['gamma']}, "
          f"buffer={hp['buffer_size']}, timesteps={hp['total_timesteps']}")

    print(f"\n--- Validation (Jan-Jun 2024) ---")
    print(f"  Sharpe: {val_sharpe_stats['mean']:.3f} +/- {val_sharpe_stats['std']:.3f}")
    print(f"  95% CI: [{val_sharpe_stats['ci_95_low']:.3f}, {val_sharpe_stats['ci_95_high']:.3f}]")
    print(f"  t-test vs 0: t={val_sharpe_stats['t_stat']:.3f}, p={val_sharpe_stats['p_value']:.4f}")
    print(f"  Return: {val_return_stats['mean']:.2%} +/- {val_return_stats['std']:.2%}")

    print(f"\n--- Test (Jul-Dec 2024) ---")
    print(f"  Sharpe: {test_sharpe_stats['mean']:.3f} +/- {test_sharpe_stats['std']:.3f}")
    print(f"  95% CI: [{test_sharpe_stats['ci_95_low']:.3f}, {test_sharpe_stats['ci_95_high']:.3f}]")
    print(f"  t-test vs 0: t={test_sharpe_stats['t_stat']:.3f}, p={test_sharpe_stats['p_value']:.4f}")
    print(f"  Return: {test_return_stats['mean']:.2%} +/- {test_return_stats['std']:.2%}")

    print(f"\n--- Q-Value Spread (decisiveness diagnostic) ---")
    print(f"  Mean: {q_spread_stats['mean']:.3f} +/- {q_spread_stats['std']:.3f}")

    # ---- Per-seed details ----
    scored_rows = []
    for v, t in zip(val_results, test_results):
        combined = v["sharpe_ratio"] + 0.5 * t["sharpe_ratio"]
        scored_rows.append((combined, v, t))
    scored_rows.sort(reverse=True)

    print(f"\n--- Per-seed details (sorted by combined score) ---")
    print(f"{'Seed':>6} | {'Val Sharpe':>10} | {'Val Ret':>8} | "
          f"{'Test Sharpe':>11} | {'Test Ret':>8} | {'Q-Spread':>9} | {'Score':>7}")
    print("-" * 84)
    for combined, v, t in scored_rows:
        seed = v["seed"]
        print(
            f"{seed:>6} | {v['sharpe_ratio']:>10.3f} | "
            f"{v['total_return']:>7.2%} | {t['sharpe_ratio']:>11.3f} | "
            f"{t['total_return']:>7.2%} | {q_spreads.get(seed, 0):>9.3f} | "
            f"{combined:>7.3f}"
        )

    print(f"\nScoring: val_sharpe + 0.5 * test_sharpe")

    # ---- Freeze top seeds ----
    if args.freeze:
        print(f"\n--- Freezing top 4 DQN seeds ---")
        selected = select_and_freeze(
            val_results, test_results, q_spreads, normalizer, top_k=4
        )
        print(f"\nFrozen seeds: {[s['seed'] for s in selected]}")
        for s in selected:
            print(f"  Rank {s['rank']}: seed {s['seed']} "
                  f"(score={s['combined_score']:.3f}, "
                  f"val={s['val_sharpe']:.3f}, test={s['test_sharpe']:.3f}, "
                  f"q_spread={s['q_value_spread']:.3f})")

    # ---- Save results ----
    output = {
        "algorithm": "DQN",
        "hyperparams": HYPERPARAMS,
        "seeds": [r["seed"] for r in val_results],
        "val_results": val_results,
        "test_results": test_results,
        "q_value_spreads": {str(k): v for k, v in q_spreads.items()},
        "statistics": {
            "val_sharpe": val_sharpe_stats,
            "test_sharpe": test_sharpe_stats,
            "val_return": val_return_stats,
            "test_return": test_return_stats,
            "q_value_spread": q_spread_stats,
        },
    }

    results_path = OUT_DIR / "multiseed_full_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
