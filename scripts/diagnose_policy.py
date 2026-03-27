"""Diagnose PPO policy distributions after Stage 1 quick test.

Loads each trained seed from multiseed_es, runs through the test split,
extracts raw policy distributions (temperature=1.0) at each timestep,
and reports entropy ratio, mean max probability, % steps with max > 0.5,
and action distribution.

Usage:
    PYTHONPATH=. python3 -m scripts.diagnose_policy
    PYTHONPATH=. python3 -m scripts.diagnose_policy --seeds 42 123 456 789 999
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.executor.env_factory import make_trading_env
from src.executor.policy import get_policy_distribution, load_executor

logger = logging.getLogger(__name__)

OUT_DIR = Path("experiments/executor/multiseed_v6")


def diagnose_seed(seed: int, split: str = "test") -> dict:
    """Run a single seed through the split and extract policy distributions."""
    seed_dir = OUT_DIR / f"seed_{seed}" / "best"
    if not (seed_dir / "model.zip").exists():
        logger.warning(f"Seed {seed}: no checkpoint found at {seed_dir}")
        return {}

    model, vec_normalize = load_executor(seed_dir)

    # Create eval env — deterministic, full split
    env_fn = make_trading_env(split=split, random_start=False)
    env = env_fn()

    obs, _ = env.reset()
    all_probs = []
    actions_taken = []

    max_steps = 10_000
    for _ in range(max_steps):
        # Get raw policy distribution (no temperature scaling)
        probs = get_policy_distribution(
            model, obs, vec_normalize=vec_normalize, temperature=1.0
        )
        all_probs.append(probs)

        # Take deterministic action (argmax)
        action = int(np.argmax(probs))
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()

    probs_arr = np.array(all_probs)  # (T, 3)
    n_steps = len(probs_arr)

    # Entropy ratio: H(pi) / log(3)
    log3 = np.log(3)
    # Clip to avoid log(0)
    eps = 1e-10
    entropies = -np.sum(probs_arr * np.log(probs_arr + eps), axis=1)
    entropy_ratios = entropies / log3

    # Max probability per timestep
    max_probs = np.max(probs_arr, axis=1)

    # Action distribution
    n_actions = len(actions_taken)
    pct_flat = actions_taken.count(0) / max(n_actions, 1)
    pct_long = actions_taken.count(1) / max(n_actions, 1)
    pct_short = actions_taken.count(2) / max(n_actions, 1)

    # Mean probabilities per action
    mean_p_flat = float(np.mean(probs_arr[:, 0]))
    mean_p_long = float(np.mean(probs_arr[:, 1]))
    mean_p_short = float(np.mean(probs_arr[:, 2]))

    result = {
        "seed": seed,
        "n_steps": n_steps,
        "mean_entropy_ratio": round(float(np.mean(entropy_ratios)), 4),
        "min_entropy_ratio": round(float(np.min(entropy_ratios)), 4),
        "max_entropy_ratio": round(float(np.max(entropy_ratios)), 4),
        "mean_max_prob": round(float(np.mean(max_probs)), 4),
        "min_max_prob": round(float(np.min(max_probs)), 4),
        "max_max_prob": round(float(np.max(max_probs)), 4),
        "pct_steps_max_gt_0.5": round(float(np.mean(max_probs > 0.5)) * 100, 1),
        "pct_steps_max_gt_0.6": round(float(np.mean(max_probs > 0.6)) * 100, 1),
        "pct_steps_max_gt_0.7": round(float(np.mean(max_probs > 0.7)) * 100, 1),
        "pct_flat": round(pct_flat * 100, 1),
        "pct_long": round(pct_long * 100, 1),
        "pct_short": round(pct_short * 100, 1),
        "mean_p_flat": round(mean_p_flat, 4),
        "mean_p_long": round(mean_p_long, 4),
        "mean_p_short": round(mean_p_short, 4),
    }

    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Diagnose PPO policy distributions")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999],
        help="Seeds to diagnose"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Data split to evaluate on"
    )
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("PPO POLICY DISTRIBUTION DIAGNOSTIC — Stage 2 Log-Return Reward")
    print("=" * 90)

    results = []
    for seed in args.seeds:
        result = diagnose_seed(seed, split=args.split)
        if not result:
            continue
        results.append(result)

        print(f"\n--- Seed {seed} ({result['n_steps']} steps on {args.split}) ---")
        print(f"  Entropy ratio:  mean={result['mean_entropy_ratio']:.4f}  "
              f"min={result['min_entropy_ratio']:.4f}  max={result['max_entropy_ratio']:.4f}")
        print(f"  Max probability: mean={result['mean_max_prob']:.4f}  "
              f"min={result['min_max_prob']:.4f}  max={result['max_max_prob']:.4f}")
        print(f"  % steps max>0.5: {result['pct_steps_max_gt_0.5']:.1f}%  "
              f">0.6: {result['pct_steps_max_gt_0.6']:.1f}%  "
              f">0.7: {result['pct_steps_max_gt_0.7']:.1f}%")
        print(f"  Action dist (argmax): flat={result['pct_flat']:.1f}%  "
              f"long={result['pct_long']:.1f}%  short={result['pct_short']:.1f}%")
        print(f"  Mean probs: p(flat)={result['mean_p_flat']:.4f}  "
              f"p(long)={result['mean_p_long']:.4f}  p(short)={result['mean_p_short']:.4f}")

    if not results:
        print("\nNo seeds found to diagnose.")
        return

    # Aggregate summary
    print("\n" + "=" * 90)
    print("AGGREGATE SUMMARY")
    print("=" * 90)

    mean_ent = np.mean([r["mean_entropy_ratio"] for r in results])
    mean_maxp = np.mean([r["mean_max_prob"] for r in results])
    mean_gt05 = np.mean([r["pct_steps_max_gt_0.5"] for r in results])

    print(f"  Mean entropy ratio across seeds: {mean_ent:.4f}")
    print(f"  Mean max probability across seeds: {mean_maxp:.4f}")
    print(f"  Mean % steps with max>0.5: {mean_gt05:.1f}%")

    # Success criteria
    print("\n--- SUCCESS CRITERIA ---")
    ent_pass = mean_ent < 0.9
    maxp_pass = mean_gt05 > 25.0  # at least 25% of steps
    print(f"  Entropy ratio < 0.9:        {'PASS' if ent_pass else 'FAIL'} ({mean_ent:.4f})")
    print(f"  >25% steps with max>0.5:    {'PASS' if maxp_pass else 'FAIL'} ({mean_gt05:.1f}%)")

    if ent_pass and maxp_pass:
        print("\n  >>> STAGE 2 PASSED — proceed to freeze top seeds <<<")
    else:
        print("\n  >>> STAGE 2 FAILED — further investigation needed <<<")

    # Save results
    output_path = OUT_DIR / "stage2_diagnostic.json"
    with open(output_path, "w") as f:
        json.dump({
            "split": args.split,
            "seeds": args.seeds,
            "per_seed": results,
            "aggregate": {
                "mean_entropy_ratio": round(float(mean_ent), 4),
                "mean_max_prob": round(float(mean_maxp), 4),
                "mean_pct_steps_max_gt_0.5": round(float(mean_gt05), 1),
            },
            "passed": bool(ent_pass and maxp_pass),
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
