"""Run baseline strategies for ablation comparison with the full Trinity system.

Baselines share the same portfolio tracking / return calculation / output
format as ``cgate_integration.py`` so that metrics are directly comparable.

Three baselines are implemented:
  1. **Executor-only** — PPO argmax at full position, no C-Gate, no Guardian.
  2. **Analyst-only** — GPT-5 signal executed directly, no RL.
  3. **Trinity-no-CGate** — If argmax(π_RL)==d_LLM → full position;
                           if disagree → argmax(π_RL) at 50% position.
                           No Δ, no thresholds, no Guardian.

Usage::

    python scripts/run_baselines.py --baseline executor-only --split test
    python scripts/run_baselines.py --baseline analyst-only --split both
    python scripts/run_baselines.py --baseline trinity-no-cgate --seed 789
    python scripts/run_baselines.py --baseline all --split both
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.cgate.divergence import ACTION_MAP
from src.executor.env_factory import make_trading_env
from src.executor.evaluate import compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio
from src.executor.policy import get_policy_distribution, load_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INITIAL_PORTFOLIO_VALUE = 100_000.0
DEFAULT_SIGNALS_PATH = "data/processed/precomputed_signals_gpt5.json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_precomputed_signals(
    signals_path: str = DEFAULT_SIGNALS_PATH,
) -> dict[str, str]:
    """Load precomputed Analyst signals, returning date -> decision mapping."""
    with open(signals_path, "r") as f:
        signals_cache = json.load(f)
    return {entry["date"]: entry["decision"] for entry in signals_cache.values()}


def _resolve_action_index(decision: str) -> int:
    """Map an Analyst decision string to an action index."""
    return ACTION_MAP[decision.lower().strip()]


def _compute_statistics(
    results: list[dict],
    *,
    baseline: str,
    model_dir: str | None,
    split: str,
    extra: dict | None = None,
) -> dict:
    """Compute standard statistics from a list of per-step result dicts."""
    n = len(results)
    port_returns = np.array([r["step_return"] for r in results])
    cumulative = np.cumprod(1.0 + port_returns)
    total_return = float(cumulative[-1] - 1.0) if len(cumulative) > 0 else 0.0
    sharpe = compute_sharpe_ratio(port_returns)
    sortino = compute_sortino_ratio(port_returns)
    max_dd = compute_max_drawdown(cumulative)
    final_pv = results[-1]["portfolio_value"] if results else INITIAL_PORTFOLIO_VALUE

    stats: dict = {
        "baseline": baseline,
        "model_dir": str(model_dir) if model_dir else None,
        "split": split,
        "n_timesteps": n,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "final_portfolio_value": final_pv,
    }
    if extra:
        stats.update(extra)
    return stats


def _track_portfolio_step(
    current_price: float,
    new_price: float,
    effective_position: float,
    portfolio_value: float,
    peak_value: float,
) -> tuple[float, float, float, float]:
    """Compute one step of portfolio tracking.

    Returns:
        (step_return, new_portfolio_value, new_peak_value, daily_pnl)
    """
    if current_price > 0 and new_price > 0:
        price_return = (new_price - current_price) / current_price
        step_return = effective_position * price_return
    else:
        step_return = 0.0

    daily_pnl = step_return * portfolio_value
    portfolio_value *= (1.0 + step_return)
    if portfolio_value > peak_value:
        peak_value = portfolio_value

    return step_return, portfolio_value, peak_value, daily_pnl


# ---------------------------------------------------------------------------
# Baseline 1: Executor-only
# ---------------------------------------------------------------------------


def run_executor_only(
    model_dir: str,
    split: str = "test",
) -> dict:
    """Run PPO Executor with argmax action at full position — no C-Gate, no Guardian.

    Args:
        model_dir: Path to frozen Executor model directory.
        split: Data split ("val" or "test").

    Returns:
        Dict with statistics and per-step results.
    """
    logger.info(f"[Executor-Only] Loading model from {model_dir}")
    model, vec_normalize = load_executor(model_dir)

    env_fn = make_trading_env(split=split, random_start=False, episode_length=None)
    env = env_fn()
    obs, info = env.reset()
    done = False

    results: list[dict] = []
    portfolio_value = INITIAL_PORTFOLIO_VALUE
    peak_value = INITIAL_PORTFOLIO_VALUE
    effective_position = 0.0

    while not done:
        date = info.get("date", "")
        current_price = info.get("price", 0.0)

        # Get policy distribution (no temperature scaling — raw logits)
        pi_rl = get_policy_distribution(model, obs, vec_normalize, temperature=1.0)
        action = int(np.argmax(pi_rl))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        # Full position — scale = 1.0
        new_effective_position = float(env_position)

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        results.append({
            "date": date,
            "action": action,
            "rl_argmax": action,
            "pi_rl": pi_rl.tolist(),
            "position": env_position,
            "effective_position": new_effective_position,
            "position_scale": 1.0,
            "price": new_price,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
            "reward": float(reward),
        })

        effective_position = new_effective_position

    env.close()

    # Action distribution
    actions = Counter(r["action"] for r in results)
    n = len(results)
    extra = {
        "pct_flat": actions.get(0, 0) / n if n else 0,
        "pct_long": actions.get(1, 0) / n if n else 0,
        "pct_short": actions.get(2, 0) / n if n else 0,
    }

    stats = _compute_statistics(
        results, baseline="executor-only", model_dir=model_dir, split=split, extra=extra
    )
    logger.info(
        f"[Executor-Only] {split} | Sharpe: {stats['sharpe_ratio']:.4f} | "
        f"Return: {stats['total_return']:.4%} | MaxDD: {stats['max_drawdown']:.4%}"
    )
    return {"statistics": stats, "results": results}


# ---------------------------------------------------------------------------
# Baseline 2: Analyst-only
# ---------------------------------------------------------------------------


def run_analyst_only(
    split: str = "test",
    signals_path: str = DEFAULT_SIGNALS_PATH,
) -> dict:
    """Execute LLM decisions directly — no RL, no C-Gate, no Guardian.

    Args:
        split: Data split ("val" or "test").
        signals_path: Path to precomputed Analyst signals JSON.

    Returns:
        Dict with statistics and per-step results.
    """
    logger.info(f"[Analyst-Only] Loading signals from {signals_path}")
    date_to_decision = load_precomputed_signals(signals_path)
    logger.info(f"[Analyst-Only] Loaded {len(date_to_decision)} signals.")

    env_fn = make_trading_env(split=split, random_start=False, episode_length=None)
    env = env_fn()
    obs, info = env.reset()
    done = False

    results: list[dict] = []
    portfolio_value = INITIAL_PORTFOLIO_VALUE
    peak_value = INITIAL_PORTFOLIO_VALUE
    effective_position = 0.0
    missing_signal_dates: list[str] = []

    while not done:
        date = info.get("date", "")
        current_price = info.get("price", 0.0)

        # Look up LLM signal
        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            action = _resolve_action_index(d_llm)
            has_signal = True
        else:
            # No signal → default to flat
            d_llm = "hold"
            action = 0
            has_signal = False
            if date:
                missing_signal_dates.append(date)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = float(env_position)

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        results.append({
            "date": date,
            "action": action,
            "analyst_decision": d_llm,
            "has_analyst_signal": has_signal,
            "position": env_position,
            "effective_position": new_effective_position,
            "position_scale": 1.0,
            "price": new_price,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
            "reward": float(reward),
        })

        effective_position = new_effective_position

    env.close()

    # Signal / action distribution
    decisions = Counter(r["analyst_decision"] for r in results)
    n = len(results)
    extra = {
        "n_missing_signals": len(missing_signal_dates),
        "signal_distribution": dict(decisions),
        "pct_hold": decisions.get("hold", 0) / n if n else 0,
        "pct_buy": decisions.get("buy", 0) / n if n else 0,
        "pct_sell": decisions.get("sell", 0) / n if n else 0,
    }

    stats = _compute_statistics(
        results, baseline="analyst-only", model_dir=None, split=split, extra=extra
    )
    logger.info(
        f"[Analyst-Only] {split} | Sharpe: {stats['sharpe_ratio']:.4f} | "
        f"Return: {stats['total_return']:.4%} | MaxDD: {stats['max_drawdown']:.4%}"
    )
    return {"statistics": stats, "results": results}


# ---------------------------------------------------------------------------
# Baseline 3: Trinity-no-CGate
# ---------------------------------------------------------------------------


def run_trinity_no_cgate(
    model_dir: str,
    split: str = "test",
    signals_path: str = DEFAULT_SIGNALS_PATH,
    disagree_scale: float = 0.5,
) -> dict:
    """Trinity without the C-Gate: agree → full position, disagree → 50%.

    Uses a simple binary agreement check between argmax(π_RL) and d_LLM
    instead of computing Δ. No thresholds, no temperature, no Guardian.

    Args:
        model_dir: Path to frozen Executor model directory.
        split: Data split ("val" or "test").
        signals_path: Path to precomputed Analyst signals JSON.
        disagree_scale: Position scale when agents disagree (default 0.5).

    Returns:
        Dict with statistics and per-step results.
    """
    logger.info(f"[Trinity-no-CGate] Loading model from {model_dir}")
    model, vec_normalize = load_executor(model_dir)

    logger.info(f"[Trinity-no-CGate] Loading signals from {signals_path}")
    date_to_decision = load_precomputed_signals(signals_path)

    env_fn = make_trading_env(split=split, random_start=False, episode_length=None)
    env = env_fn()
    obs, info = env.reset()
    done = False

    results: list[dict] = []
    portfolio_value = INITIAL_PORTFOLIO_VALUE
    peak_value = INITIAL_PORTFOLIO_VALUE
    effective_position = 0.0
    missing_signal_dates: list[str] = []

    while not done:
        date = info.get("date", "")
        current_price = info.get("price", 0.0)

        # Get RL argmax (no temperature scaling)
        pi_rl = get_policy_distribution(model, obs, vec_normalize, temperature=1.0)
        rl_action = int(np.argmax(pi_rl))

        # Get Analyst decision
        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            llm_action = _resolve_action_index(d_llm)
            has_signal = True
        else:
            d_llm = None
            llm_action = -1  # sentinel — no signal
            has_signal = False
            if date:
                missing_signal_dates.append(date)

        # Agreement logic
        if has_signal and rl_action == llm_action:
            # Agree → full conviction
            action = rl_action
            position_scale = 1.0
            agreement = "agree"
        else:
            # Disagree (or missing signal) → execute RL argmax at reduced size
            action = rl_action
            position_scale = disagree_scale
            agreement = "disagree"

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = env_position * position_scale

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        results.append({
            "date": date,
            "action": action,
            "rl_argmax": rl_action,
            "analyst_decision": d_llm if has_signal else "MISSING",
            "llm_action": llm_action if has_signal else None,
            "has_analyst_signal": has_signal,
            "agreement": agreement,
            "position_scale": position_scale,
            "position": env_position,
            "effective_position": new_effective_position,
            "pi_rl": pi_rl.tolist(),
            "price": new_price,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
            "reward": float(reward),
        })

        effective_position = new_effective_position

    env.close()

    # Agreement statistics
    n = len(results)
    n_agree = sum(1 for r in results if r["agreement"] == "agree")
    n_disagree = n - n_agree

    extra = {
        "n_missing_signals": len(missing_signal_dates),
        "disagree_scale": disagree_scale,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "pct_agree": n_agree / n if n else 0,
        "pct_disagree": n_disagree / n if n else 0,
    }

    stats = _compute_statistics(
        results, baseline="trinity-no-cgate", model_dir=model_dir, split=split, extra=extra
    )
    logger.info(
        f"[Trinity-no-CGate] {split} | Sharpe: {stats['sharpe_ratio']:.4f} | "
        f"Return: {stats['total_return']:.4%} | MaxDD: {stats['max_drawdown']:.4%} | "
        f"Agree: {extra['pct_agree']:.1%}"
    )
    return {"statistics": stats, "results": results}


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------


def _discover_seeds(frozen_dir: Path) -> list[int]:
    """Discover valid seeds from selection.json."""
    selection_path = frozen_dir / "selection.json"
    if selection_path.exists():
        with open(selection_path) as f:
            selection = json.load(f)
        seeds = sorted(entry["seed"] for entry in selection.get("selected", []))
        logger.info(f"Using {len(seeds)} seeds from selection.json: {seeds}")
        return seeds
    # Fallback: glob seed directories
    return sorted(int(d.name.split("_")[1]) for d in frozen_dir.glob("seed_*"))


def run_all_baselines(
    baselines: list[str],
    splits: list[str],
    seeds: list[int] | None = None,
    signals_path: str = DEFAULT_SIGNALS_PATH,
    output_dir: str = "experiments/baselines",
) -> dict[str, dict]:
    """Run specified baselines across splits and seeds.

    Args:
        baselines: List of baseline names to run.
        splits: List of splits ("val", "test").
        seeds: Seeds to use. None → discover from selection.json.
        signals_path: Path to precomputed signals.
        output_dir: Where to save result JSONs.

    Returns:
        Dict mapping run_key → statistics dict.
    """
    frozen_dir = Path("experiments/executor/frozen")
    if seeds is None:
        seeds = _discover_seeds(frozen_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}

    for baseline in baselines:
        for split in splits:
            if baseline == "analyst-only":
                # No seed variation — single run per split
                key = f"analyst_only_{split}"
                logger.info(f"\n{'─' * 50}")
                logger.info(f"Running: {key}")
                logger.info(f"{'─' * 50}")

                output = run_analyst_only(split=split, signals_path=signals_path)
                all_stats[key] = output["statistics"]

                path = out / f"{key}.json"
                with open(path, "w") as f:
                    json.dump(output, f, indent=2)
                logger.info(f"Saved to {path}")
            else:
                # Seed-dependent baselines
                for seed in seeds:
                    model_dir = str(frozen_dir / f"seed_{seed}")
                    if not Path(model_dir).exists():
                        logger.warning(f"Model dir not found: {model_dir}, skipping.")
                        continue

                    tag = baseline.replace("-", "_")
                    key = f"{tag}_{split}_seed{seed}"
                    logger.info(f"\n{'─' * 50}")
                    logger.info(f"Running: {key}")
                    logger.info(f"{'─' * 50}")

                    if baseline == "executor-only":
                        output = run_executor_only(model_dir=model_dir, split=split)
                    elif baseline == "trinity-no-cgate":
                        output = run_trinity_no_cgate(
                            model_dir=model_dir, split=split, signals_path=signals_path
                        )
                    else:
                        raise ValueError(f"Unknown baseline: {baseline}")

                    all_stats[key] = output["statistics"]

                    path = out / f"{key}.json"
                    with open(path, "w") as f:
                        json.dump(output, f, indent=2)
                    logger.info(f"Saved to {path}")

    # ── Summary table ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 80)
    header = f"{'Run':<40s} {'Sharpe':>8s} {'Sortino':>8s} {'Return':>9s} {'MaxDD':>8s}"
    logger.info(header)
    logger.info("-" * len(header))
    for key, stats in all_stats.items():
        logger.info(
            f"{key:<40s} "
            f"{stats['sharpe_ratio']:>8.4f} "
            f"{stats['sortino_ratio']:>8.4f} "
            f"{stats['total_return']:>8.4%} "
            f"{stats['max_drawdown']:>8.4%}"
        )

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Run baseline strategies")
    parser.add_argument(
        "--baseline",
        default="all",
        choices=["executor-only", "analyst-only", "trinity-no-cgate", "all"],
        help="Which baseline to run (default: all)",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["val", "test", "both"],
        help="Data split (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed (default: all from selection.json)",
    )
    parser.add_argument(
        "--signals-path",
        default=DEFAULT_SIGNALS_PATH,
        help="Path to precomputed Analyst signals JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/baselines",
        help="Output directory for results (default: experiments/baselines)",
    )
    args = parser.parse_args()

    if args.baseline == "all":
        baselines = ["executor-only", "analyst-only", "trinity-no-cgate"]
    else:
        baselines = [args.baseline]

    splits = ["val", "test"] if args.split == "both" else [args.split]
    seeds = [args.seed] if args.seed is not None else None

    run_all_baselines(
        baselines=baselines,
        splits=splits,
        seeds=seeds,
        signals_path=args.signals_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
