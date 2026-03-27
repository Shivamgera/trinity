"""Threshold calibration for the Consistency Gate.

Computes the empirical Δ distribution on validation data across all frozen
Executor seeds, then selects (τ_low, τ_high) at configurable percentiles.

Calibration on the validation split is standard hyperparameter tuning and
avoids information leakage from the test split.

Usage (standalone)::

    python -m src.cgate.calibrate --temperature 0.01

Programmatic::

    from src.cgate.calibrate import calibrate_thresholds
    result = calibrate_thresholds(temperature=0.01)
    print(result["tau_low"], result["tau_high"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.cgate.divergence import compute_delta
from src.executor.env_factory import make_trading_env
from src.executor.policy import get_policy_distribution, load_executor

logger = logging.getLogger(__name__)


def collect_val_deltas(
    frozen_dir: str | Path = "experiments/executor/frozen",
    signals_path: str | Path = "data/processed/precomputed_signals.json",
    temperature: float = 1.0,
    seeds: Sequence[int] | None = None,
) -> np.ndarray:
    """Collect Δ values on the validation split across frozen seeds.

    Args:
        frozen_dir: Directory containing frozen seed subdirectories.
        signals_path: Path to precomputed Analyst signals JSON.
        temperature: Softmax temperature for policy distribution extraction.
        seeds: Specific seeds to use.  If None, reads from selection.json
               (which lists only the top-K validated seeds).

    Returns:
        1-D array of Δ values pooled across all seeds.
    """
    frozen_dir = Path(frozen_dir)

    # Load precomputed signals
    with open(signals_path, "r") as f:
        signals = json.load(f)
    date_to_decision = {s["date"]: s["decision"] for s in signals.values()}

    # Discover seeds from selection.json (only validated seeds)
    if seeds is None:
        selection_path = frozen_dir / "selection.json"
        if selection_path.exists():
            with open(selection_path, "r") as f:
                selection = json.load(f)
            seeds = [entry["seed"] for entry in selection["selected"]]
            logger.info(f"Using {len(seeds)} seeds from selection.json: {seeds}")
        else:
            seed_dirs = sorted(frozen_dir.glob("seed_*"))
            seeds = [int(d.name.split("_")[1]) for d in seed_dirs]
    if not seeds:
        raise FileNotFoundError(f"No frozen seeds found in {frozen_dir}")

    all_deltas: list[float] = []

    for seed in seeds:
        model_dir = frozen_dir / f"seed_{seed}"
        logger.info(f"Loading seed {seed} from {model_dir}")
        model, vec_norm = load_executor(str(model_dir))

        env_fn = make_trading_env(split="val", random_start=False, episode_length=None)
        env = env_fn()
        obs, info = env.reset()
        done = False

        while not done:
            date = info.get("date", "")
            d = date_to_decision.get(date)

            pi = get_policy_distribution(model, obs, vec_norm, temperature=temperature)

            if d is not None:
                delta = compute_delta(d, pi)
            else:
                delta = 1.0  # Missing signal → treat as conflict

            all_deltas.append(delta)

            obs, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated
        env.close()

    return np.array(all_deltas)


def calibrate_thresholds(
    temperature: float = 1.0,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
    frozen_dir: str | Path = "experiments/executor/frozen",
    signals_path: str | Path = "data/processed/precomputed_signals.json",
    seeds: Sequence[int] | None = None,
) -> dict:
    """Calibrate C-Gate thresholds from the empirical validation-set Δ distribution.

    Picks τ_low at the *low_percentile* of the pooled Δ distribution and τ_high at
    the *high_percentile*.  This ensures that, on validation data, roughly
    ``low_percentile`` % of timesteps fall in the agreement regime and
    ``(100 - high_percentile)`` % fall in the conflict regime.

    Args:
        temperature: Softmax temperature for policy extraction.
        low_percentile: Percentile for τ_low (default 20 → ~20% agreement).
        high_percentile: Percentile for τ_high (default 80 → ~20% conflict).
        frozen_dir: Directory containing frozen seed subdirectories.
        signals_path: Path to precomputed Analyst signals JSON.
        seeds: Specific seeds to use.

    Returns:
        Dict with calibration results including tau_low, tau_high, and diagnostics.
    """
    deltas = collect_val_deltas(
        frozen_dir=frozen_dir,
        signals_path=signals_path,
        temperature=temperature,
        seeds=seeds,
    )

    tau_low = float(np.percentile(deltas, low_percentile))
    tau_high = float(np.percentile(deltas, high_percentile))

    # Compute regime distribution with these thresholds
    n = len(deltas)
    n_agree = int(np.sum(deltas <= tau_low))
    n_ambig = int(np.sum((deltas > tau_low) & (deltas <= tau_high)))
    n_conflict = int(np.sum(deltas > tau_high))

    result = {
        "temperature": temperature,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "tau_low": round(tau_low, 4),
        "tau_high": round(tau_high, 4),
        "n_samples": n,
        "delta_mean": round(float(deltas.mean()), 4),
        "delta_std": round(float(deltas.std()), 4),
        "delta_median": round(float(np.median(deltas)), 4),
        "delta_p10": round(float(np.percentile(deltas, 10)), 4),
        "delta_p25": round(float(np.percentile(deltas, 25)), 4),
        "delta_p75": round(float(np.percentile(deltas, 75)), 4),
        "delta_p90": round(float(np.percentile(deltas, 90)), 4),
        "pct_agreement": round(n_agree / n, 4) if n else 0,
        "pct_ambiguity": round(n_ambig / n, 4) if n else 0,
        "pct_conflict": round(n_conflict / n, 4) if n else 0,
    }

    logger.info(f"Calibration results (T={temperature}):")
    logger.info(f"  τ_low  = {result['tau_low']:.4f}  (p{low_percentile:.0f})")
    logger.info(f"  τ_high = {result['tau_high']:.4f}  (p{high_percentile:.0f})")
    logger.info(f"  Agreement:  {result['pct_agreement']:.1%}")
    logger.info(f"  Ambiguity:  {result['pct_ambiguity']:.1%}")
    logger.info(f"  Conflict:   {result['pct_conflict']:.1%}")

    return result
