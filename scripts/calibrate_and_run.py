"""Calibrate C-Gate thresholds on validation data and run integration.

Two-step workflow:
  1. Calibrate τ_low, τ_high from the empirical Δ distribution (val split).
  2. Run the full C-Gate integration on val and/or test with calibrated thresholds.

Usage::

    # Calibrate + run integration on test (all seeds, T=0.01)
    python scripts/calibrate_and_run.py

    # Custom percentiles and single seed
    python scripts/calibrate_and_run.py --low-pct 15 --high-pct 85 --seed 123

    # Calibrate only (skip integration)
    python scripts/calibrate_and_run.py --calibrate-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.cgate.calibrate import calibrate_thresholds
from scripts.cgate_integration import run_cgate_integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate C-Gate thresholds and run integration"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (default: 1.0)",
    )
    parser.add_argument(
        "--low-pct",
        type=float,
        default=20.0,
        help="Percentile for τ_low (default: 20)",
    )
    parser.add_argument(
        "--high-pct",
        type=float,
        default=80.0,
        help="Percentile for τ_high (default: 80)",
    )
    parser.add_argument(
        "--signals-path",
        default="data/processed/precomputed_signals.json",
        help="Path to precomputed Analyst signals JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to run integration on (default: all frozen seeds)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "both"],
        help="Split(s) to run integration on (default: test)",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only calibrate thresholds; skip integration",
    )
    parser.add_argument(
        "--no-guardian",
        action="store_true",
        help="Disable Guardian (no position scaling, no stop-loss, no circuit breakers)",
    )
    args = parser.parse_args()

    # ── Step 1: Calibrate ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Threshold calibration on validation split")
    logger.info("=" * 60)

    cal = calibrate_thresholds(
        temperature=args.temperature,
        low_percentile=args.low_pct,
        high_percentile=args.high_pct,
        signals_path=args.signals_path,
    )

    tau_low = cal["tau_low"]
    tau_high = cal["tau_high"]

    # Save calibration result
    output_dir = Path("experiments/cgate")
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_path = output_dir / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    logger.info(f"Calibration saved to {cal_path}")

    if args.calibrate_only:
        logger.info("--calibrate-only set; skipping integration.")
        return

    # ── Step 2: Run integration ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: C-Gate integration with calibrated thresholds")
    logger.info(f"  τ_low={tau_low:.4f}, τ_high={tau_high:.4f}, T={args.temperature}")
    logger.info("=" * 60)

    frozen_dir = Path("experiments/executor/frozen")
    if args.seed is not None:
        seeds = [args.seed]
    else:
        # Use selection.json to avoid incompatible seeds (1024, 4096)
        selection_path = frozen_dir / "selection.json"
        if selection_path.exists():
            with open(selection_path) as f:
                selection = json.load(f)
            seeds = sorted(
                entry["seed"] for entry in selection.get("selected", [])
            )
            logger.info(f"Using {len(seeds)} seeds from selection.json: {seeds}")
        else:
            seeds = sorted(
                int(d.name.split("_")[1]) for d in frozen_dir.glob("seed_*")
            )

    splits = ["val", "test"] if args.split == "both" else [args.split]

    all_results = {}
    for seed in seeds:
        model_dir = str(frozen_dir / f"seed_{seed}")
        for split in splits:
            key = f"seed_{seed}_{split}"
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Running: seed={seed}, split={split}")
            logger.info(f"{'─' * 40}")

            output = run_cgate_integration(
                model_dir=model_dir,
                split=split,
                tau_low=tau_low,
                tau_high=tau_high,
                signals_path=args.signals_path,
                temperature=args.temperature,
                enable_guardian=not args.no_guardian,
            )

            stats = output["statistics"]
            all_results[key] = stats

            # Save per-run results
            run_path = output_dir / f"integration_{split}_seed{seed}_calibrated.json"
            with open(run_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved to {run_path}")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: Calibrated C-Gate Integration")
    logger.info(f"Thresholds: τ_low={tau_low:.4f}, τ_high={tau_high:.4f}, T={args.temperature}")
    logger.info("=" * 60)
    header = f"{'Run':<25s} {'Agree':>7s} {'Ambig':>7s} {'Confl':>7s} {'Sharpe':>8s} {'Return':>9s} {'MaxDD':>8s}"
    logger.info(header)
    logger.info("-" * len(header))
    for key, stats in all_results.items():
        logger.info(
            f"{key:<25s} "
            f"{stats['pct_agreement']:>6.1%} "
            f"{stats['pct_ambiguity']:>6.1%} "
            f"{stats['pct_conflict']:>6.1%} "
            f"{stats['sharpe_ratio']:>8.4f} "
            f"{stats['total_return']:>8.4%} "
            f"{stats['max_drawdown']:>8.4%}"
        )


if __name__ == "__main__":
    main()
