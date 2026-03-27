"""Hyperparameter sweep for C-Gate temperature and threshold percentiles.

Strategy:
  1. For each temperature, collect val Δ distribution once (~10s per temp).
  2. For each (low_pct, high_pct) pair, derive (τ_low, τ_high) from the
     cached Δ distribution (instant — just numpy percentile calls).
  3. Run the full Guardian-enabled integration on val split for all 4 seeds.
  4. Rank by mean val Sharpe (or median for robustness).
  5. Report top-N configurations.

Usage::

    python scripts/sweep_cgate.py
    python scripts/sweep_cgate.py --top 10
    python scripts/sweep_cgate.py --temperature 0.01  # single temperature
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np

from src.cgate.calibrate import collect_val_deltas
from scripts.cgate_integration import run_cgate_integration

logging.basicConfig(level=logging.WARNING)  # Quiet by default
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Sweep grid
DEFAULT_TEMPERATURES = [0.005, 0.01, 0.02, 0.05, 0.1]
DEFAULT_LOW_PCTS = [10, 15, 20, 25, 30]
DEFAULT_HIGH_PCTS = [70, 75, 80, 85, 90]
SEEDS = [123, 789, 2048, 7777]
SIGNALS_PATH = "data/processed/precomputed_signals_gpt5.json"
FROZEN_DIR = "experiments/executor/frozen"


def run_sweep(
    temperatures: list[float] | None = None,
    low_pcts: list[float] | None = None,
    high_pcts: list[float] | None = None,
    seeds: list[int] | None = None,
    signals_path: str = SIGNALS_PATH,
    frozen_dir: str = FROZEN_DIR,
    enable_guardian: bool = True,
    top_n: int = 20,
) -> list[dict]:
    """Run the full sweep and return ranked results.

    Args:
        temperatures: List of softmax temperatures to sweep.
        low_pcts: List of low percentiles for τ_low.
        high_pcts: List of high percentiles for τ_high.
        seeds: Executor seeds to evaluate.
        signals_path: Path to precomputed Analyst signals.
        frozen_dir: Path to frozen Executor models.
        enable_guardian: Whether to enable the Guardian pipeline.
        top_n: Number of top results to display.

    Returns:
        List of result dicts sorted by mean val Sharpe (descending).
    """
    temperatures = temperatures or DEFAULT_TEMPERATURES
    low_pcts = low_pcts or DEFAULT_LOW_PCTS
    high_pcts = high_pcts or DEFAULT_HIGH_PCTS
    seeds = seeds or SEEDS

    # Step 1: Pre-collect val Δ distributions per temperature
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting validation Δ distributions per temperature")
    logger.info("=" * 60)

    delta_cache: dict[float, np.ndarray] = {}
    for temp in temperatures:
        logger.info(f"  Collecting deltas at T={temp}...")
        t0 = time.time()
        deltas = collect_val_deltas(
            frozen_dir=frozen_dir,
            signals_path=signals_path,
            temperature=temp,
            seeds=seeds,
        )
        delta_cache[temp] = deltas
        logger.info(f"    -> {len(deltas)} deltas in {time.time() - t0:.1f}s "
                     f"(mean={deltas.mean():.4f}, std={deltas.std():.4f})")

    # Step 2: Enumerate all (temp, low_pct, high_pct) combinations
    combos = list(product(temperatures, low_pcts, high_pcts))
    n_combos = len(combos)
    n_runs = n_combos * len(seeds)  # each combo runs all seeds on val
    logger.info(f"\nSTEP 2: Sweeping {n_combos} combinations × {len(seeds)} seeds = {n_runs} runs")

    all_results: list[dict] = []
    t_start = time.time()

    for i, (temp, low_pct, high_pct) in enumerate(combos):
        # Derive thresholds from cached delta distribution
        deltas = delta_cache[temp]
        tau_low = float(np.percentile(deltas, low_pct))
        tau_high = float(np.percentile(deltas, high_pct))

        # Skip degenerate cases where thresholds are too close
        if tau_high - tau_low < 0.01:
            continue

        seed_sharpes = []
        seed_returns = []
        seed_max_dds = []

        for seed in seeds:
            model_dir = f"{frozen_dir}/seed_{seed}"
            try:
                output = run_cgate_integration(
                    model_dir=model_dir,
                    split="val",
                    tau_low=tau_low,
                    tau_high=tau_high,
                    signals_path=signals_path,
                    temperature=temp,
                    enable_guardian=enable_guardian,
                )
                stats = output["statistics"]
                seed_sharpes.append(stats["sharpe_ratio"])
                seed_returns.append(stats["total_return"])
                seed_max_dds.append(stats["max_drawdown"])
            except Exception as e:
                logger.warning(f"Failed: T={temp}, pcts=({low_pct},{high_pct}), seed={seed}: {e}")
                seed_sharpes.append(float("nan"))
                seed_returns.append(float("nan"))
                seed_max_dds.append(float("nan"))

        # Aggregate across seeds
        sharpes = np.array(seed_sharpes)
        valid = ~np.isnan(sharpes)
        if not valid.any():
            continue

        result = {
            "temperature": temp,
            "low_pct": low_pct,
            "high_pct": high_pct,
            "tau_low": round(tau_low, 4),
            "tau_high": round(tau_high, 4),
            "mean_sharpe": float(np.nanmean(sharpes)),
            "median_sharpe": float(np.nanmedian(sharpes)),
            "std_sharpe": float(np.nanstd(sharpes)),
            "min_sharpe": float(np.nanmin(sharpes)),
            "max_sharpe": float(np.nanmax(sharpes)),
            "mean_return": float(np.nanmean(seed_returns)),
            "mean_max_dd": float(np.nanmean(seed_max_dds)),
            "seed_sharpes": {s: round(sh, 4) for s, sh in zip(seeds, seed_sharpes)},
            "enable_guardian": enable_guardian,
        }
        all_results.append(result)

        # Progress
        elapsed = time.time() - t_start
        pct_done = (i + 1) / n_combos
        eta = elapsed / pct_done * (1 - pct_done) if pct_done > 0 else 0
        if (i + 1) % 5 == 0 or (i + 1) == n_combos:
            logger.info(
                f"  [{i+1}/{n_combos}] T={temp}, pcts=({low_pct},{high_pct}) -> "
                f"mean_sharpe={result['mean_sharpe']:.4f}, "
                f"mean_dd={result['mean_max_dd']:.4%} "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    # Sort by mean val Sharpe descending
    all_results.sort(key=lambda r: r["mean_sharpe"], reverse=True)

    # Print top-N
    logger.info(f"\n{'=' * 100}")
    logger.info(f"TOP {min(top_n, len(all_results))} CONFIGURATIONS (ranked by mean val Sharpe)")
    logger.info(f"{'=' * 100}")
    header = (
        f"{'Rank':>4s}  {'T':>6s}  {'Lo%':>4s}  {'Hi%':>4s}  "
        f"{'τ_low':>7s}  {'τ_high':>7s}  "
        f"{'Mean':>8s}  {'Med':>8s}  {'Std':>7s}  "
        f"{'MeanRet':>9s}  {'MeanDD':>8s}  "
        f"{'S123':>7s}  {'S789':>7s}  {'S2048':>7s}  {'S7777':>7s}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for rank, r in enumerate(all_results[:top_n], 1):
        seed_strs = [f"{r['seed_sharpes'].get(s, float('nan')):>7.3f}" for s in seeds]
        logger.info(
            f"{rank:>4d}  {r['temperature']:>6.3f}  {r['low_pct']:>4.0f}  {r['high_pct']:>4.0f}  "
            f"{r['tau_low']:>7.4f}  {r['tau_high']:>7.4f}  "
            f"{r['mean_sharpe']:>8.4f}  {r['median_sharpe']:>8.4f}  {r['std_sharpe']:>7.4f}  "
            f"{r['mean_return']:>8.4%}  {r['mean_max_dd']:>8.4%}  "
            + "  ".join(seed_strs)
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="C-Gate hyperparameter sweep")
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="*",
        default=None,
        help="Specific temperature(s) to sweep (default: full grid)",
    )
    parser.add_argument(
        "--low-pcts",
        type=float,
        nargs="*",
        default=None,
        help="Low percentiles to sweep (default: [10,15,20,25,30])",
    )
    parser.add_argument(
        "--high-pcts",
        type=float,
        nargs="*",
        default=None,
        help="High percentiles to sweep (default: [70,75,80,85,90])",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results to display (default: 20)",
    )
    parser.add_argument(
        "--no-guardian",
        action="store_true",
        help="Disable Guardian for sweep (ablation comparison)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/cgate/sweep_results.json",
        help="Output path for sweep results JSON",
    )
    args = parser.parse_args()

    results = run_sweep(
        temperatures=args.temperature,
        low_pcts=args.low_pcts,
        high_pcts=args.high_pcts,
        enable_guardian=not args.no_guardian,
        top_n=args.top,
    )

    # Save all results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll {len(results)} results saved to {output_path}")

    if results:
        best = results[0]
        logger.info(f"\nBEST CONFIG:")
        logger.info(f"  Temperature:  {best['temperature']}")
        logger.info(f"  Percentiles:  ({best['low_pct']}, {best['high_pct']})")
        logger.info(f"  Thresholds:   τ_low={best['tau_low']}, τ_high={best['tau_high']}")
        logger.info(f"  Mean Sharpe:  {best['mean_sharpe']:.4f}")
        logger.info(f"  Mean Return:  {best['mean_return']:.4%}")
        logger.info(f"  Mean MaxDD:   {best['mean_max_dd']:.4%}")


if __name__ == "__main__":
    main()
