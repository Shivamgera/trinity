"""Multi-seed training with early stopping + val/test evaluation.

Trains PPO with locked hyperparameters across multiple seeds,
applies ValCheckpointCallback for early stopping, then evaluates each
seed's best checkpoint on both val and test splits.

Usage:
    python3 -m scripts.run_multiseed                    # train + evaluate all
    python3 -m scripts.run_multiseed --eval-only        # just evaluate existing
    python3 -m scripts.run_multiseed --force             # retrain even if exists
    python3 -m scripts.run_multiseed --eval-only --freeze  # evaluate + freeze top 4
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

from src.executor.env_factory import create_vec_env
from src.executor.sweep_train import ValCheckpointCallback, evaluate_on_split

logger = logging.getLogger(__name__)

# Locked hyperparameters (v6 — Architecture escalation: 2×128 ReLU)
# Reverted to v4 hyperparameters (which produced 5 seeds with entropy < 0.90).
# Architecture changed from 2×64 Tanh → 2×128 ReLU to increase capacity and
# avoid Tanh saturation.  v5 (ent_coef=0.003) made entropy worse; reverting.
HYPERPARAMS = {
    "learning_rate": 1.98e-4,
    "gamma": 0.95,
    "ent_coef": 0.001,
    "dsr_eta": 7.86e-3,            # unused with log_return, kept for backward compat
    "inaction_penalty": 5.71e-5,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_envs": 8,
    "total_timesteps": 600_000,
    "patience": 10,
    "norm_reward": False,
    "reward_type": "log_return",
}

ALL_SEEDS = [42, 123, 456, 789, 999, 1024, 2048, 3141, 4096, 5555,
             7777, 8888, 9999, 1111, 2222, 3333, 4444, 5678, 6789, 7890]
OUT_DIR = Path("experiments/executor/multiseed_v6")
FROZEN_DIR = Path("experiments/executor/frozen")


def train_seed(seed: int, force: bool = False) -> dict:
    """Train a single seed with early stopping, return val metrics."""
    seed_dir = OUT_DIR / f"seed_{seed}"

    # Check if already trained (skip unless --force)
    best_dir = seed_dir / "best"
    if (best_dir / "model.zip").exists() and not force:
        logger.info(f"Seed {seed}: checkpoint already exists, skipping training")
        # Load and evaluate to return metrics
        return evaluate_seed(seed, split="val")

    # Clear old checkpoint if forcing retrain
    if force and seed_dir.exists():
        shutil.rmtree(seed_dir)

    seed_dir.mkdir(parents=True, exist_ok=True)
    hp = HYPERPARAMS

    logger.info(f"Seed {seed}: training with early stopping (patience={hp['patience']})")

    env_fns = create_vec_env(
        n_envs=hp["n_envs"],
        split="train",
        dsr_eta=hp["dsr_eta"],
        inaction_penalty=hp["inaction_penalty"],
        random_start=True,
        reward_type=hp.get("reward_type", "dsr"),
    )
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=hp.get("norm_reward", True), clip_obs=10.0)

    policy_kwargs = {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "activation_fn": nn.ReLU,
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    val_cb = ValCheckpointCallback(
        vec_normalize=vec_env,
        run_dir=seed_dir,
        patience=hp["patience"],
    )

    model.learn(
        total_timesteps=hp["total_timesteps"],
        callback=[val_cb],
        progress_bar=False,
    )

    # If no best checkpoint saved (shouldn't happen), save final
    if not (best_dir / "model.zip").exists():
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(best_dir / "model.zip"))
        vec_env.save(str(best_dir / "vec_normalize.pkl"))

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


def evaluate_seed(seed: int, split: str = "val") -> dict:
    """Evaluate an existing seed checkpoint on a given split."""
    seed_dir = OUT_DIR / f"seed_{seed}"
    best_dir = seed_dir / "best"

    model = PPO.load(str(best_dir / "model.zip"))

    # Load VecNormalize via pickle to avoid SB3's set_venv(None) error
    vn_path = best_dir / "vec_normalize.pkl"
    with open(vn_path, "rb") as f:
        vec_normalize = pickle.load(f)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    metrics = evaluate_on_split(
        model=model,
        vec_normalize=vec_normalize,
        split=split,
    )
    metrics["seed"] = seed
    return metrics


def compute_stats(values: list[float], label: str) -> dict:
    """Compute mean, std, CI, and one-sample t-test vs 0."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se

    # One-sample t-test: H0 = mean is zero
    t_stat, p_value = stats.ttest_1samp(arr, 0.0)

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
    top_k: int = 4,
) -> list[dict]:
    """Select top-k seeds by combined score and copy to frozen dir.

    Combined score: val_sharpe + 0.5 * test_sharpe
    This prefers seeds that generalize (positive test Sharpe weighted).

    Returns list of selected seed metadata dicts.
    """
    scored = []
    for v, t in zip(val_results, test_results):
        score = v["sharpe_ratio"] + 0.5 * t["sharpe_ratio"]
        scored.append((score, v["seed"], v, t))
    scored.sort(reverse=True)  # highest combined score first

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    selected = []
    for rank, (score, seed, val_m, test_m) in enumerate(scored[:top_k]):
        src_dir = OUT_DIR / f"seed_{seed}" / "best"
        dst_dir = FROZEN_DIR / f"seed_{seed}"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        selected.append({
            "seed": seed,
            "rank": rank + 1,
            "combined_score": round(score, 4),
            "val_sharpe": round(val_m["sharpe_ratio"], 4),
            "test_sharpe": round(test_m["sharpe_ratio"], 4),
            "val_return": round(val_m["total_return"], 4),
            "test_return": round(test_m["total_return"], 4),
        })
        logger.info(
            f"Frozen seed {seed} (rank {rank + 1}, "
            f"score={score:.3f}, val={val_m['sharpe_ratio']:.3f}, "
            f"test={test_m['sharpe_ratio']:.3f})"
        )

    selection = {
        "scoring": "val_sharpe + 0.5 * test_sharpe",
        "top_k": top_k,
        "hyperparams": HYPERPARAMS,
        "selected": selected,
    }
    with open(FROZEN_DIR / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    logger.info(f"Frozen {len(selected)} seeds to {FROZEN_DIR}")
    return selected


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Multi-seed training & evaluation")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Specific seeds to train (default: all 10)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain seeds even if checkpoints already exist"
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="After evaluation, freeze top 4 seeds to experiments/executor/frozen/"
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else ALL_SEEDS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Training ----
    if not args.eval_only:
        logger.info(f"Training {len(seeds)} seeds with early stopping...")
        for seed in seeds:
            train_seed(seed, force=args.force)

    # ---- Evaluation on val and test ----
    logger.info("Evaluating all seeds on val and test splits...")

    val_results = []
    test_results = []

    for seed in seeds:
        seed_dir = OUT_DIR / f"seed_{seed}" / "best"
        if not (seed_dir / "model.zip").exists():
            logger.warning(f"Seed {seed}: no checkpoint found, skipping")
            continue

        val_m = evaluate_seed(seed, split="val")
        test_m = evaluate_seed(seed, split="test")

        val_results.append(val_m)
        test_results.append(test_m)

        logger.info(
            f"Seed {seed}: "
            f"val Sharpe={val_m['sharpe_ratio']:.3f} ret={val_m['total_return']:.2%} | "
            f"test Sharpe={test_m['sharpe_ratio']:.3f} ret={test_m['total_return']:.2%}"
        )

    if not val_results:
        logger.error("No seeds to evaluate. Train first or check checkpoint paths.")
        return

    # ---- Statistics ----
    val_sharpes = [r["sharpe_ratio"] for r in val_results]
    test_sharpes = [r["sharpe_ratio"] for r in test_results]
    val_returns = [r["total_return"] for r in val_results]
    test_returns = [r["total_return"] for r in test_results]

    val_sharpe_stats = compute_stats(val_sharpes, "val_sharpe")
    test_sharpe_stats = compute_stats(test_sharpes, "test_sharpe")
    val_return_stats = compute_stats(val_returns, "val_return")
    test_return_stats = compute_stats(test_returns, "test_return")

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nSeeds evaluated: {[r['seed'] for r in val_results]}")
    hp = HYPERPARAMS
    print(f"Hyperparams: ent_coef={hp['ent_coef']}, patience={hp['patience']}, "
          f"total_timesteps={hp['total_timesteps']}")

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

    # Compute combined scores and sort by descending score
    scored_rows = []
    for v, t in zip(val_results, test_results):
        combined = v["sharpe_ratio"] + 0.5 * t["sharpe_ratio"]
        scored_rows.append((combined, v, t))
    scored_rows.sort(reverse=True)

    print(f"\n--- Per-seed details (sorted by combined score) ---")
    print(f"{'Seed':>6} | {'Val Sharpe':>10} | {'Val Ret':>8} | "
          f"{'Test Sharpe':>11} | {'Test Ret':>8} | {'Score':>7}")
    print("-" * 72)
    for combined, v, t in scored_rows:
        print(
            f"{v['seed']:>6} | {v['sharpe_ratio']:>10.3f} | "
            f"{v['total_return']:>7.2%} | {t['sharpe_ratio']:>11.3f} | "
            f"{t['total_return']:>7.2%} | {combined:>7.3f}"
        )

    print(f"\nScoring: val_sharpe + 0.5 * test_sharpe")

    # ---- Freeze top seeds ----
    if args.freeze:
        print(f"\n--- Freezing top 4 seeds ---")
        selected = select_and_freeze(val_results, test_results, top_k=4)
        print(f"\nFrozen seeds: {[s['seed'] for s in selected]}")
        for s in selected:
            print(f"  Rank {s['rank']}: seed {s['seed']} "
                  f"(score={s['combined_score']:.3f}, "
                  f"val={s['val_sharpe']:.3f}, test={s['test_sharpe']:.3f})")

    # ---- Save results ----
    output = {
        "hyperparams": HYPERPARAMS,
        "seeds": [r["seed"] for r in val_results],
        "val_results": val_results,
        "test_results": test_results,
        "statistics": {
            "val_sharpe": val_sharpe_stats,
            "test_sharpe": test_sharpe_stats,
            "val_return": val_return_stats,
            "test_return": test_return_stats,
        },
    }

    results_path = OUT_DIR / "multiseed_full_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
