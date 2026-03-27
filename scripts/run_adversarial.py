"""Adversarial evaluation of the Trinity architecture and baselines.

Two attack vectors are implemented:

1. **Analyst Poisoning** — Flip a fraction of directional LLM signals
   (buy↔sell). Hold signals are left unchanged. This simulates an
   adversary corrupting the Analyst's output channel.

2. **Executor Perturbation** — Add Gaussian noise N(0, σ=rate) to the
   raw observation vector before feeding it to the PPO policy. Since
   features are z-normalised, σ=0.5 means adding half a standard
   deviation of noise. This simulates adversarial state manipulation.

Four configurations are evaluated under each attack:
  - **Trinity** (best sweep: T=0.05, Guardian enabled)
  - **Executor-Only** (PPO argmax, full position, no C-Gate/Guardian)
  - **Analyst-Only** (GPT-5 signal executed directly)
  - **Trinity-no-CGate** (agree→full, disagree→50%, no Guardian)

Usage::

    python scripts/run_adversarial.py --attack analyst-poison --split test
    python scripts/run_adversarial.py --attack executor-perturb --split test
    python scripts/run_adversarial.py --attack all --split test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.cgate.divergence import ACTION_MAP
from src.cgate.gate import CGateResult, ConsistencyGate
from src.executor.env_factory import make_trading_env
from src.executor.evaluate import compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio
from src.executor.policy import get_policy_distribution, load_executor
from src.guardian import FinalAction, Guardian
from src.guardian.config import load_guardian_config
from src.guardian.hard_constraints import PortfolioState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INITIAL_PORTFOLIO_VALUE = 100_000.0
DEFAULT_SIGNALS_PATH = "data/processed/precomputed_signals_gpt5.json"
CORRUPTION_SEED = 42  # Fixed seed for reproducible signal corruption / noise


# ── Best sweep config (locked) ──────────────────────────────────────────
BEST_TEMPERATURE = 1.0
BEST_TAU_LOW = 0.6935
BEST_TAU_HIGH = 0.9877


# ─────────────────────────────────────────────────────────────────────────
# Signal corruption helpers
# ─────────────────────────────────────────────────────────────────────────


def poison_signals(
    date_to_decision: dict[str, str],
    corruption_rate: float,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Flip a fraction of *directional* signals (buy↔sell). Hold unchanged.

    Args:
        date_to_decision: Clean date→decision mapping.
        corruption_rate: Fraction of directional signals to flip (0.0–1.0).
        rng: Numpy random generator for reproducibility.

    Returns:
        New dict with corrupted signals.
    """
    corrupted = dict(date_to_decision)

    # Identify directional signals
    directional_dates = [
        d for d, dec in corrupted.items() if dec.lower() in ("buy", "sell")
    ]

    n_to_flip = int(len(directional_dates) * corruption_rate)
    if n_to_flip == 0:
        return corrupted

    flip_dates = set(rng.choice(directional_dates, size=n_to_flip, replace=False))

    for date in flip_dates:
        dec = corrupted[date].lower()
        corrupted[date] = "sell" if dec == "buy" else "buy"

    return corrupted


# ─────────────────────────────────────────────────────────────────────────
# Shared portfolio tracking
# ─────────────────────────────────────────────────────────────────────────


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
    portfolio_value *= 1.0 + step_return
    if portfolio_value > peak_value:
        peak_value = portfolio_value

    return step_return, portfolio_value, peak_value, daily_pnl


def _compute_statistics(
    results: list[dict],
    *,
    config: str,
    attack: str,
    corruption_rate: float,
    split: str,
    seed: int | None,
    extra: dict | None = None,
) -> dict:
    """Compute standard statistics from per-step results."""
    n = len(results)
    port_returns = np.array([r["step_return"] for r in results])
    cumulative = np.cumprod(1.0 + port_returns)
    total_return = float(cumulative[-1] - 1.0) if len(cumulative) > 0 else 0.0
    sharpe = compute_sharpe_ratio(port_returns)
    sortino = compute_sortino_ratio(port_returns)
    max_dd = compute_max_drawdown(cumulative)
    final_pv = results[-1]["portfolio_value"] if results else INITIAL_PORTFOLIO_VALUE

    stats: dict = {
        "config": config,
        "attack": attack,
        "corruption_rate": corruption_rate,
        "split": split,
        "seed": seed,
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


# ─────────────────────────────────────────────────────────────────────────
# Signal loading
# ─────────────────────────────────────────────────────────────────────────


def _load_signals(path: str = DEFAULT_SIGNALS_PATH) -> dict[str, str]:
    with open(path, "r") as f:
        signals_cache = json.load(f)
    return {entry["date"]: entry["decision"] for entry in signals_cache.values()}


def _resolve_action(decision: str) -> int:
    return ACTION_MAP[decision.lower().strip()]


# ─────────────────────────────────────────────────────────────────────────
# Config runners (4 configurations)
# ─────────────────────────────────────────────────────────────────────────


def _run_trinity(
    model_dir: str,
    split: str,
    date_to_decision: dict[str, str],
    obs_noise_rng: np.random.Generator | None = None,
    obs_noise_sigma: float = 0.0,
) -> list[dict]:
    """Run full Trinity (C-Gate + Guardian) with optional obs noise."""
    model, vec_normalize = load_executor(model_dir)
    gate = ConsistencyGate(tau_low=BEST_TAU_LOW, tau_high=BEST_TAU_HIGH)

    hard_config, adaptive_config = load_guardian_config("configs/guardian.yaml")
    guardian = Guardian(hard_config=hard_config, adaptive_config=adaptive_config)

    env_fn = make_trading_env(split=split, random_start=False, episode_length=None)
    env = env_fn()
    obs, info = env.reset()
    done = False

    results: list[dict] = []
    portfolio_value = INITIAL_PORTFOLIO_VALUE
    peak_value = INITIAL_PORTFOLIO_VALUE
    cash = INITIAL_PORTFOLIO_VALUE
    daily_pnl = 0.0
    effective_position = 0.0
    entry_price = 0.0
    stop_loss_threshold: float | None = None

    while not done:
        date = info.get("date", "")
        current_price = info.get("price", 0.0)

        # Stop-loss check
        stop_loss_triggered = False
        if (
            stop_loss_threshold is not None
            and effective_position != 0.0
            and entry_price > 0.0
            and current_price > 0.0
        ):
            if effective_position > 0:
                unrealized_return = (current_price - entry_price) / entry_price
            else:
                unrealized_return = (entry_price - current_price) / entry_price
            if unrealized_return < -stop_loss_threshold:
                stop_loss_triggered = True

        # Get policy distribution (with optional obs noise)
        obs_for_policy = obs.copy()
        if obs_noise_rng is not None and obs_noise_sigma > 0:
            noise = obs_noise_rng.normal(0.0, obs_noise_sigma, size=obs.shape)
            obs_for_policy = obs + noise.astype(obs.dtype)

        pi_rl = get_policy_distribution(
            model, obs_for_policy, vec_normalize, temperature=BEST_TEMPERATURE
        )

        # Get Analyst decision
        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            has_signal = True
        else:
            d_llm = None
            has_signal = False

        # C-Gate
        if has_signal:
            cgate_result = gate.evaluate(d_llm, pi_rl)
        else:
            cgate_result = CGateResult(delta=1.0, regime="conflict", action=0)

        # Guardian
        guardian_action = cgate_result.action
        position_scale = 1.0
        stop_loss_override: float | None = None
        blocked_by_stage1 = False

        if stop_loss_triggered:
            guardian_action = 0
            position_scale = 0.0
        else:
            current_drawdown = (
                (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
            )
            portfolio_state = PortfolioState(
                position=effective_position,
                cash=cash,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                peak_value=peak_value,
                current_drawdown=current_drawdown,
            )
            final_action: FinalAction = guardian.process(
                proposed_action=cgate_result.action,
                portfolio_state=portfolio_state,
                cgate_result=cgate_result,
            )
            guardian_action = final_action.action
            position_scale = final_action.position_scale
            stop_loss_override = final_action.stop_loss
            blocked_by_stage1 = final_action.blocked_by_stage1

        obs, reward, terminated, truncated, info = env.step(guardian_action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = env_position * position_scale

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        cash = portfolio_value * (1.0 - abs(new_effective_position))

        # Update entry price / stop-loss
        if new_effective_position != 0.0 and effective_position == 0.0:
            entry_price = new_price
        elif new_effective_position == 0.0:
            entry_price = 0.0

        effective_position = new_effective_position
        stop_loss_threshold = stop_loss_override

        results.append({
            "date": date,
            "regime": cgate_result.regime,
            "action": guardian_action,
            "position_scale": position_scale,
            "effective_position": effective_position,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
        })

    env.close()
    return results


def _run_executor_only(
    model_dir: str,
    split: str,
    obs_noise_rng: np.random.Generator | None = None,
    obs_noise_sigma: float = 0.0,
) -> list[dict]:
    """Run Executor-Only baseline with optional obs noise."""
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

        # Policy distribution with optional noise
        obs_for_policy = obs.copy()
        if obs_noise_rng is not None and obs_noise_sigma > 0:
            noise = obs_noise_rng.normal(0.0, obs_noise_sigma, size=obs.shape)
            obs_for_policy = obs + noise.astype(obs.dtype)

        pi_rl = get_policy_distribution(
            model, obs_for_policy, vec_normalize, temperature=1.0
        )
        action = int(np.argmax(pi_rl))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = float(env_position)

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        effective_position = new_effective_position

        results.append({
            "date": date,
            "action": action,
            "effective_position": effective_position,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
        })

    env.close()
    return results


def _run_analyst_only(
    split: str,
    date_to_decision: dict[str, str],
) -> list[dict]:
    """Run Analyst-Only baseline (no obs noise applies)."""
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

        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            action = _resolve_action(d_llm)
        else:
            action = 0  # flat

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = float(env_position)

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        effective_position = new_effective_position

        results.append({
            "date": date,
            "action": action,
            "effective_position": effective_position,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
        })

    env.close()
    return results


def _run_trinity_no_cgate(
    model_dir: str,
    split: str,
    date_to_decision: dict[str, str],
    obs_noise_rng: np.random.Generator | None = None,
    obs_noise_sigma: float = 0.0,
    disagree_scale: float = 0.5,
) -> list[dict]:
    """Run Trinity-no-CGate baseline with optional obs noise."""
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

        # Policy with optional noise
        obs_for_policy = obs.copy()
        if obs_noise_rng is not None and obs_noise_sigma > 0:
            noise = obs_noise_rng.normal(0.0, obs_noise_sigma, size=obs.shape)
            obs_for_policy = obs + noise.astype(obs.dtype)

        pi_rl = get_policy_distribution(
            model, obs_for_policy, vec_normalize, temperature=1.0
        )
        rl_action = int(np.argmax(pi_rl))

        # Analyst decision
        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            llm_action = _resolve_action(d_llm)
            has_signal = True
        else:
            llm_action = -1
            has_signal = False

        # Agreement logic
        if has_signal and rl_action == llm_action:
            action = rl_action
            position_scale = 1.0
        else:
            action = rl_action
            position_scale = disagree_scale

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env_position = info.get("position", 0)
        new_price = info.get("price", 0.0)
        new_effective_position = env_position * position_scale

        step_return, portfolio_value, peak_value, daily_pnl = _track_portfolio_step(
            current_price, new_price, effective_position, portfolio_value, peak_value
        )

        effective_position = new_effective_position

        results.append({
            "date": date,
            "action": action,
            "position_scale": position_scale,
            "effective_position": effective_position,
            "step_return": step_return,
            "portfolio_value": portfolio_value,
        })

    env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────
# Attack orchestrator
# ─────────────────────────────────────────────────────────────────────────


def run_analyst_poisoning(
    seeds: list[int],
    split: str,
    corruption_rates: list[float],
    signals_path: str = DEFAULT_SIGNALS_PATH,
    output_dir: str = "experiments/adversarial",
) -> list[dict]:
    """Run Analyst Poisoning attack across all configs, seeds, and rates."""
    frozen_dir = Path("experiments/executor/frozen")
    out = Path(output_dir) / "analyst_poison"
    out.mkdir(parents=True, exist_ok=True)

    clean_signals = _load_signals(signals_path)
    all_stats: list[dict] = []

    for rate in corruption_rates:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ANALYST POISONING — corruption_rate={rate:.0%}")
        logger.info(f"{'=' * 60}")

        # Create corrupted signals for this rate (deterministic)
        rng = np.random.default_rng(CORRUPTION_SEED)
        corrupted_signals = poison_signals(clean_signals, rate, rng)

        # Count flips
        n_flipped = sum(
            1 for d in clean_signals
            if clean_signals[d] != corrupted_signals.get(d, clean_signals[d])
        )
        logger.info(f"  Flipped {n_flipped} directional signals")

        # ── Analyst-Only (single run, no seed) ──
        logger.info(f"  [Analyst-Only] rate={rate:.0%}")
        results = _run_analyst_only(split, corrupted_signals)
        stats = _compute_statistics(
            results, config="analyst-only", attack="analyst-poison",
            corruption_rate=rate, split=split, seed=None,
            extra={"n_flipped": n_flipped},
        )
        all_stats.append(stats)
        _save(out, f"analyst_only_rate{int(rate*100)}.json", stats, results)
        logger.info(
            f"    Sharpe={stats['sharpe_ratio']:.4f} | "
            f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
        )

        # ── Seed-dependent configs ──
        for seed in seeds:
            model_dir = str(frozen_dir / f"seed_{seed}")

            # Trinity (best sweep)
            logger.info(f"  [Trinity] seed={seed} rate={rate:.0%}")
            results = _run_trinity(model_dir, split, corrupted_signals)
            stats = _compute_statistics(
                results, config="trinity", attack="analyst-poison",
                corruption_rate=rate, split=split, seed=seed,
            )
            all_stats.append(stats)
            _save(out, f"trinity_seed{seed}_rate{int(rate*100)}.json", stats, results)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

            # Executor-Only (unaffected by signal corruption — run once at rate 0 only)
            # Executor-Only does not use signals, so results are identical at all rates.
            # We still record them for completeness at each rate.
            if rate == corruption_rates[0]:
                logger.info(f"  [Executor-Only] seed={seed} (unaffected by signal poison)")
                results = _run_executor_only(model_dir, split)
                stats = _compute_statistics(
                    results, config="executor-only", attack="analyst-poison",
                    corruption_rate=0.0, split=split, seed=seed,
                )
                # Save at rate 0 — reuse for all rates
                _save(out, f"executor_only_seed{seed}.json", stats, results)
                # Record stats for each rate with identical values
                for r in corruption_rates:
                    s = dict(stats)
                    s["corruption_rate"] = r
                    all_stats.append(s)
                logger.info(
                    f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                    f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
                )

            # Trinity-no-CGate
            logger.info(f"  [Trinity-no-CGate] seed={seed} rate={rate:.0%}")
            results = _run_trinity_no_cgate(model_dir, split, corrupted_signals)
            stats = _compute_statistics(
                results, config="trinity-no-cgate", attack="analyst-poison",
                corruption_rate=rate, split=split, seed=seed,
            )
            all_stats.append(stats)
            _save(out, f"trinity_no_cgate_seed{seed}_rate{int(rate*100)}.json", stats, results)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

    return all_stats


def run_executor_perturbation(
    seeds: list[int],
    split: str,
    corruption_rates: list[float],
    signals_path: str = DEFAULT_SIGNALS_PATH,
    output_dir: str = "experiments/adversarial",
) -> list[dict]:
    """Run Executor Perturbation attack across all configs, seeds, and rates."""
    frozen_dir = Path("experiments/executor/frozen")
    out = Path(output_dir) / "executor_perturb"
    out.mkdir(parents=True, exist_ok=True)

    clean_signals = _load_signals(signals_path)
    all_stats: list[dict] = []

    for rate in corruption_rates:
        sigma = rate  # Direct mapping: σ = rate
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EXECUTOR PERTURBATION — σ={sigma:.2f}")
        logger.info(f"{'=' * 60}")

        # ── Analyst-Only (unaffected by obs noise — run once at rate 0) ──
        if rate == corruption_rates[0]:
            logger.info(f"  [Analyst-Only] (unaffected by obs perturbation)")
            results = _run_analyst_only(split, clean_signals)
            stats = _compute_statistics(
                results, config="analyst-only", attack="executor-perturb",
                corruption_rate=0.0, split=split, seed=None,
            )
            _save(out, "analyst_only.json", stats, results)
            for r in corruption_rates:
                s = dict(stats)
                s["corruption_rate"] = r
                all_stats.append(s)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

        for seed in seeds:
            model_dir = str(frozen_dir / f"seed_{seed}")

            # Each (seed, rate) combo gets a fresh noise RNG so noise is
            # reproducible per-run but different across seeds/rates.
            noise_rng = np.random.default_rng(CORRUPTION_SEED + seed + int(rate * 1000))

            # Trinity (best sweep)
            logger.info(f"  [Trinity] seed={seed} σ={sigma:.2f}")
            noise_rng_trinity = np.random.default_rng(CORRUPTION_SEED + seed + int(rate * 1000))
            results = _run_trinity(
                model_dir, split, clean_signals,
                obs_noise_rng=noise_rng_trinity, obs_noise_sigma=sigma,
            )
            stats = _compute_statistics(
                results, config="trinity", attack="executor-perturb",
                corruption_rate=rate, split=split, seed=seed,
            )
            all_stats.append(stats)
            _save(out, f"trinity_seed{seed}_sigma{int(rate*100)}.json", stats, results)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

            # Executor-Only
            logger.info(f"  [Executor-Only] seed={seed} σ={sigma:.2f}")
            noise_rng_exec = np.random.default_rng(CORRUPTION_SEED + seed + int(rate * 1000))
            results = _run_executor_only(
                model_dir, split,
                obs_noise_rng=noise_rng_exec, obs_noise_sigma=sigma,
            )
            stats = _compute_statistics(
                results, config="executor-only", attack="executor-perturb",
                corruption_rate=rate, split=split, seed=seed,
            )
            all_stats.append(stats)
            _save(out, f"executor_only_seed{seed}_sigma{int(rate*100)}.json", stats, results)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

            # Trinity-no-CGate
            logger.info(f"  [Trinity-no-CGate] seed={seed} σ={sigma:.2f}")
            noise_rng_tnc = np.random.default_rng(CORRUPTION_SEED + seed + int(rate * 1000))
            results = _run_trinity_no_cgate(
                model_dir, split, clean_signals,
                obs_noise_rng=noise_rng_tnc, obs_noise_sigma=sigma,
            )
            stats = _compute_statistics(
                results, config="trinity-no-cgate", attack="executor-perturb",
                corruption_rate=rate, split=split, seed=seed,
            )
            all_stats.append(stats)
            _save(out, f"trinity_no_cgate_seed{seed}_sigma{int(rate*100)}.json", stats, results)
            logger.info(
                f"    Sharpe={stats['sharpe_ratio']:.4f} | "
                f"Return={stats['total_return']:.4%} | MaxDD={stats['max_drawdown']:.4%}"
            )

    return all_stats


# ─────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────


def _save(
    out_dir: Path,
    filename: str,
    stats: dict,
    results: list[dict],
) -> None:
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump({"statistics": stats, "results": results}, f, indent=2)


def _discover_seeds(frozen_dir: Path) -> list[int]:
    selection_path = frozen_dir / "selection.json"
    if selection_path.exists():
        with open(selection_path) as f:
            selection = json.load(f)
        seeds = sorted(entry["seed"] for entry in selection.get("selected", []))
        logger.info(f"Using {len(seeds)} seeds from selection.json: {seeds}")
        return seeds
    return sorted(int(d.name.split("_")[1]) for d in frozen_dir.glob("seed_*"))


# ─────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────


def print_summary(all_stats: list[dict], attack_name: str) -> None:
    """Print a summary table grouped by config and corruption rate."""
    from collections import defaultdict

    # Group: (config, rate) → list of stats
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for s in all_stats:
        grouped[(s["config"], s["corruption_rate"])].append(s)

    configs = sorted(set(s["config"] for s in all_stats))
    rates = sorted(set(s["corruption_rate"] for s in all_stats))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"ADVERSARIAL SUMMARY — {attack_name}")
    logger.info(f"{'=' * 80}")

    # Sharpe table
    logger.info(f"\n  Mean Sharpe by corruption rate:")
    header = f"  {'Config':<22s}" + "".join(f"{'rate='+str(int(r*100))+'%':>12s}" for r in rates)
    logger.info(header)
    logger.info("  " + "-" * (22 + 12 * len(rates)))
    for cfg in configs:
        row = f"  {cfg:<22s}"
        for r in rates:
            entries = grouped.get((cfg, r), [])
            if entries:
                mean_sharpe = np.mean([e["sharpe_ratio"] for e in entries])
                row += f"{mean_sharpe:>12.4f}"
            else:
                row += f"{'N/A':>12s}"
        logger.info(row)

    # MaxDD table
    logger.info(f"\n  Mean MaxDD by corruption rate:")
    header = f"  {'Config':<22s}" + "".join(f"{'rate='+str(int(r*100))+'%':>12s}" for r in rates)
    logger.info(header)
    logger.info("  " + "-" * (22 + 12 * len(rates)))
    for cfg in configs:
        row = f"  {cfg:<22s}"
        for r in rates:
            entries = grouped.get((cfg, r), [])
            if entries:
                mean_maxdd = np.mean([e["max_drawdown"] for e in entries])
                row += f"{mean_maxdd:>11.2%} "
            else:
                row += f"{'N/A':>12s}"
        logger.info(row)

    # Sortino table
    logger.info(f"\n  Mean Sortino by corruption rate:")
    header = f"  {'Config':<22s}" + "".join(f"{'rate='+str(int(r*100))+'%':>12s}" for r in rates)
    logger.info(header)
    logger.info("  " + "-" * (22 + 12 * len(rates)))
    for cfg in configs:
        row = f"  {cfg:<22s}"
        for r in rates:
            entries = grouped.get((cfg, r), [])
            if entries:
                mean_sortino = np.mean([e["sortino_ratio"] for e in entries])
                row += f"{mean_sortino:>12.4f}"
            else:
                row += f"{'N/A':>12s}"
        logger.info(row)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Adversarial evaluation")
    parser.add_argument(
        "--attack",
        default="all",
        choices=["analyst-poison", "executor-perturb", "all"],
        help="Attack type (default: all)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Data split (default: test)",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
        help="Corruption rates (default: 0.10 0.20 0.30 0.40 0.50)",
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
        default="experiments/adversarial",
        help="Output directory (default: experiments/adversarial)",
    )
    args = parser.parse_args()

    frozen_dir = Path("experiments/executor/frozen")
    seeds = [args.seed] if args.seed is not None else _discover_seeds(frozen_dir)

    attacks = (
        ["analyst-poison", "executor-perturb"]
        if args.attack == "all"
        else [args.attack]
    )

    combined_stats: dict[str, list[dict]] = {}

    for attack in attacks:
        if attack == "analyst-poison":
            stats = run_analyst_poisoning(
                seeds=seeds,
                split=args.split,
                corruption_rates=args.rates,
                signals_path=args.signals_path,
                output_dir=args.output_dir,
            )
            combined_stats["analyst-poison"] = stats
            print_summary(stats, "Analyst Poisoning")

        elif attack == "executor-perturb":
            stats = run_executor_perturbation(
                seeds=seeds,
                split=args.split,
                corruption_rates=args.rates,
                signals_path=args.signals_path,
                output_dir=args.output_dir,
            )
            combined_stats["executor-perturb"] = stats
            print_summary(stats, "Executor Perturbation")

    # Save combined summary
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "adversarial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined_stats, f, indent=2)
    logger.info(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
