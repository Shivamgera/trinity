"""
C-Gate integration test: Run full Trinity simulation on val/test splits.

This script runs the Executor through TradingEnv while synchronizing
with precomputed Analyst signals via date matching.  At each step the
C-Gate arbitrates between the two channels and the Guardian enforces
risk limits, producing a realistic trading trajectory.

Pipeline per step:
  1. env.reset()/step() -> obs, info["date"]
  2. date -> date_to_signal[date] -> d_LLM
  3. obs -> get_policy_distribution(model, obs, vec_normalize) -> pi_RL
  4. gate.evaluate(d_LLM, pi_RL) -> CGateResult(delta, regime, action)
  5. Guardian Stage 1: hard constraints (circuit breakers)
  6. Guardian Stage 2: adaptive policy (position scaling, stop-loss)
  7. Execute final action in env; apply position_scale in return calc

Usage:
    python scripts/cgate_integration.py
    python scripts/cgate_integration.py --seed 789
    python scripts/cgate_integration.py --split val
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.cgate.gate import CGateResult, ConsistencyGate
from src.executor.env_factory import make_trading_env
from src.executor.evaluate import compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio
from src.executor.policy import get_policy_distribution, load_executor
from src.guardian import Guardian, FinalAction
from src.guardian.hard_constraints import PortfolioState
from src.guardian.config import load_guardian_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_precomputed_signals(
    signals_path: str = "data/processed/precomputed_signals.json",
) -> dict[str, str]:
    """Load precomputed Analyst signals, returning date -> decision mapping."""
    with open(signals_path, "r") as f:
        signals_cache = json.load(f)

    date_to_decision: dict[str, str] = {}
    for entry in signals_cache.values():
        date_to_decision[entry["date"]] = entry["decision"]

    return date_to_decision


def run_cgate_integration(
    model_dir: str = "experiments/executor/frozen/seed_7777",
    split: str = "test",
    tau_low: float = 0.1,
    tau_high: float = 0.4,
    signals_path: str = "data/processed/precomputed_signals.json",
    temperature: float = 1.0,
    enable_guardian: bool = True,
    guardian_config_path: str = "configs/guardian.yaml",
) -> dict:
    """Run C-Gate integration on a single split with a single frozen seed.

    This runs the Executor through TradingEnv, looks up the precomputed
    Analyst signal for each date, evaluates the C-Gate, passes the result
    through the Guardian (hard constraints + adaptive policy), and applies
    the Guardian's position scaling in the portfolio return calculation.

    Args:
        model_dir: Path to frozen Executor model directory.
        split: Data split to evaluate on ("val" or "test").
        tau_low: C-Gate lower threshold (agreement/ambiguity boundary).
        tau_high: C-Gate upper threshold (ambiguity/conflict boundary).
        signals_path: Path to precomputed Analyst signals JSON.
        temperature: Softmax temperature for policy distribution extraction.
                     Values < 1.0 sharpen the distribution.  Default 1.0.
        enable_guardian: If True, run the Guardian pipeline (Stage 1 + Stage 2).
                         If False, behave like the old script (no position scaling,
                         no stop-loss, no circuit breakers).
        guardian_config_path: Path to Guardian YAML config file.

    Returns:
        Dict with statistics and per-step results.
    """
    # Load Executor model + VecNormalize
    logger.info(f"Loading Executor from {model_dir}")
    model, vec_normalize = load_executor(model_dir)

    # Load precomputed Analyst signals
    logger.info(f"Loading precomputed signals from {signals_path}")
    date_to_decision = load_precomputed_signals(signals_path)
    logger.info(f"Loaded {len(date_to_decision)} Analyst signals.")

    # Create TradingEnv
    env_fn = make_trading_env(split=split, random_start=False, episode_length=None)
    env = env_fn()

    # Initialize C-Gate
    gate = ConsistencyGate(tau_low=tau_low, tau_high=tau_high)

    # Initialize Guardian
    guardian: Guardian | None = None
    if enable_guardian:
        try:
            hard_config, adaptive_config = load_guardian_config(guardian_config_path)
            guardian = Guardian(hard_config=hard_config, adaptive_config=adaptive_config)
            logger.info("Guardian initialized (Stage 1: hard constraints, Stage 2: adaptive policy)")
        except FileNotFoundError:
            logger.warning(f"Guardian config not found at {guardian_config_path}; running without Guardian")
            guardian = None

    # Run the C-Gate integration loop
    obs, info = env.reset()
    done = False
    results: list[dict] = []
    missing_signal_dates: list[str] = []

    # ── Portfolio tracking state for Guardian + return calculation ──
    INITIAL_PORTFOLIO_VALUE = 100_000.0  # notional starting capital
    portfolio_value = INITIAL_PORTFOLIO_VALUE
    peak_value = INITIAL_PORTFOLIO_VALUE
    cash = INITIAL_PORTFOLIO_VALUE  # start fully in cash
    daily_pnl = 0.0
    effective_position = 0.0  # position * position_scale (fractional)
    entry_price = 0.0  # for stop-loss tracking
    stop_loss_threshold: float | None = None  # current stop-loss, or None
    n_stop_loss_exits = 0
    n_stage1_blocks = 0

    while not done:
        date = info.get("date", "")
        current_price = info.get("price", 0.0)

        # ── Check stop-loss before anything else ──
        # If we have a stop-loss active and unrealized loss exceeds it,
        # override to flat for this step.
        stop_loss_triggered = False
        if (
            stop_loss_threshold is not None
            and effective_position != 0.0
            and entry_price > 0.0
            and current_price > 0.0
        ):
            # Unrealized return on the effective position direction
            if effective_position > 0:
                unrealized_return = (current_price - entry_price) / entry_price
            else:
                unrealized_return = (entry_price - current_price) / entry_price
            if unrealized_return < -stop_loss_threshold:
                stop_loss_triggered = True
                n_stop_loss_exits += 1

        # Get Executor's policy distribution (temperature-scaled)
        pi_rl = get_policy_distribution(model, obs, vec_normalize, temperature=temperature)

        # Get Analyst decision for this date
        if date and date in date_to_decision:
            d_llm = date_to_decision[date]
            has_signal = True
        else:
            # No Analyst signal -> treat as conflict (Δ=1.0), go flat
            d_llm = None
            has_signal = False
            if date:
                missing_signal_dates.append(date)

        # Run C-Gate
        if has_signal:
            cgate_result = gate.evaluate(d_llm, pi_rl)  # type: ignore[arg-type]
        else:
            # Missing signal -> forced conflict, go flat
            cgate_result = CGateResult(delta=1.0, regime="conflict", action=0)

        # ── Guardian pipeline ──
        guardian_action = cgate_result.action
        position_scale = 1.0
        stop_loss_override: float | None = None
        blocked_by_stage1 = False
        guardian_log: str | None = None

        if stop_loss_triggered:
            # Stop-loss fires unconditionally — override to flat
            guardian_action = 0
            position_scale = 0.0
            stop_loss_override = None
            guardian_log = (
                f"[STOP-LOSS] Exiting to flat: unrealized loss exceeded "
                f"{stop_loss_threshold:.1%} threshold"
            )
        elif guardian is not None:
            # Build PortfolioState for Guardian
            current_drawdown = (
                (peak_value - portfolio_value) / peak_value
                if peak_value > 0
                else 0.0
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
            if blocked_by_stage1:
                n_stage1_blocks += 1
            if final_action.stage2_result is not None:
                guardian_log = final_action.stage2_result.log_entry

        # Execute the action in the env (discrete action — env sees 0/1/2)
        obs, reward, terminated, truncated, info = env.step(guardian_action)
        done = terminated or truncated

        # ── Compute effective return with position scaling ──
        # The env returns a position in {-1, 0, 1}. We apply the Guardian's
        # position_scale to model fractional sizing in the return calc.
        env_position = info.get("position", 0)
        new_effective_position = env_position * position_scale
        new_price = info.get("price", 0.0)

        # Compute the daily portfolio return for this step
        if current_price > 0 and new_price > 0:
            price_return = (new_price - current_price) / current_price
            step_return = effective_position * price_return
        else:
            step_return = 0.0

        # ── Update portfolio tracking ──
        daily_pnl = step_return * portfolio_value
        portfolio_value *= (1.0 + step_return)
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        # Simplified cash model: cash = portfolio_value when flat,
        # otherwise cash = portfolio_value * (1 - abs(effective_position))
        cash = portfolio_value * (1.0 - abs(new_effective_position))

        # ── Update entry price and stop-loss for next step ──
        if new_effective_position != 0.0 and effective_position == 0.0:
            # Entering a new position
            entry_price = new_price
        elif new_effective_position == 0.0:
            # Flat — reset entry price
            entry_price = 0.0
        # If still in a position (same direction), keep the original entry_price

        effective_position = new_effective_position
        stop_loss_threshold = stop_loss_override

        results.append(
            {
                "date": date,
                "delta": cgate_result.delta,
                "regime": cgate_result.regime,
                "cgate_action": cgate_result.action,
                "guardian_action": guardian_action,
                "action": guardian_action,  # final executed action
                "position_scale": position_scale,
                "effective_position": effective_position,
                "stop_loss_triggered": stop_loss_triggered,
                "blocked_by_stage1": blocked_by_stage1,
                "guardian_log": guardian_log,
                "analyst_decision": d_llm if has_signal else "MISSING",
                "has_analyst_signal": has_signal,
                "rl_argmax": int(np.argmax(pi_rl)),
                "pi_rl": pi_rl.tolist(),
                "reward": float(reward),
                "position": env_position,
                "price": new_price,
                "step_return": step_return,
                "portfolio_value": portfolio_value,
                "total_pnl": info.get("total_pnl", 0.0),
            }
        )

    env.close()
    logger.info(f"Processed {len(results)} timesteps on {split} split.")
    logger.info(f"Missing Analyst signals on {len(missing_signal_dates)} dates.")

    # Compute statistics
    deltas = np.array([r["delta"] for r in results])
    regimes = Counter(r["regime"] for r in results)
    n = len(results)

    # Portfolio returns come from our position-scaled calculation
    port_returns = np.array([r["step_return"] for r in results])

    cumulative = np.cumprod(1.0 + port_returns)
    total_return = float(cumulative[-1] - 1.0) if len(cumulative) > 0 else 0.0
    sharpe = compute_sharpe_ratio(port_returns)
    sortino = compute_sortino_ratio(port_returns)
    max_dd = compute_max_drawdown(cumulative)

    # Guardian-specific stats
    guardian_stats: dict = {}
    if enable_guardian and guardian is not None:
        n_scaled = sum(1 for r in results if r["position_scale"] < 1.0 and r["position_scale"] > 0.0)
        n_stop_losses = sum(1 for r in results if r["stop_loss_triggered"])
        n_stage1 = sum(1 for r in results if r["blocked_by_stage1"])
        guardian_stats = {
            "n_position_scaled": n_scaled,
            "pct_position_scaled": n_scaled / n if n > 0 else 0,
            "n_stop_loss_exits": n_stop_losses,
            "pct_stop_loss_exits": n_stop_losses / n if n > 0 else 0,
            "n_stage1_blocks": n_stage1,
            "pct_stage1_blocks": n_stage1 / n if n > 0 else 0,
        }

    stats = {
        "model_dir": str(model_dir),
        "split": split,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "temperature": temperature,
        "enable_guardian": enable_guardian,
        "n_timesteps": n,
        "n_missing_signals": len(missing_signal_dates),
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "delta_std": float(deltas.std()),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "delta_p25": float(np.percentile(deltas, 25)),
        "delta_p75": float(np.percentile(deltas, 75)),
        "regimes": dict(regimes),
        "pct_agreement": regimes.get("agreement", 0) / n if n > 0 else 0,
        "pct_ambiguity": regimes.get("ambiguity", 0) / n if n > 0 else 0,
        "pct_conflict": regimes.get("conflict", 0) / n if n > 0 else 0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "final_portfolio_value": portfolio_value,
        **guardian_stats,
    }

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"C-GATE INTEGRATION RESULTS ({n} timesteps, {split})")
    logger.info(f"{'=' * 60}")
    logger.info(f"Model: {model_dir}")
    logger.info(f"Thresholds: tau_low={tau_low}, tau_high={tau_high}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Guardian: {'ENABLED' if enable_guardian and guardian else 'DISABLED'}")
    logger.info(f"\nDelta statistics:")
    logger.info(f"  Mean:   {stats['delta_mean']:.4f}")
    logger.info(f"  Median: {stats['delta_median']:.4f}")
    logger.info(f"  Std:    {stats['delta_std']:.4f}")
    logger.info(f"  Min:    {stats['delta_min']:.4f}")
    logger.info(f"  Max:    {stats['delta_max']:.4f}")
    logger.info(f"\nRegime distribution:")
    for regime in ["agreement", "ambiguity", "conflict"]:
        count = regimes.get(regime, 0)
        pct = count / n * 100 if n > 0 else 0
        logger.info(f"  {regime:12s}: {count:4d} ({pct:5.1f}%)")
    logger.info(f"\nTrading performance (C-Gate + Guardian controlled):")
    logger.info(f"  Sharpe ratio:  {sharpe:.4f}")
    logger.info(f"  Sortino ratio: {sortino:.4f}")
    logger.info(f"  Total return:  {total_return:.4%}")
    logger.info(f"  Max drawdown:  {max_dd:.4%}")
    if guardian_stats:
        logger.info(f"\nGuardian interventions:")
        logger.info(f"  Position scaled (ambiguity): {guardian_stats['n_position_scaled']} ({guardian_stats['pct_position_scaled']:.1%})")
        logger.info(f"  Stop-loss exits:             {guardian_stats['n_stop_loss_exits']} ({guardian_stats['pct_stop_loss_exits']:.1%})")
        logger.info(f"  Stage 1 circuit breakers:    {guardian_stats['n_stage1_blocks']} ({guardian_stats['pct_stage1_blocks']:.1%})")

    # Print examples for each regime
    logger.info(f"\nExample outputs per regime:")
    for regime in ["agreement", "ambiguity", "conflict"]:
        examples = [r for r in results if r["regime"] == regime]
        if examples:
            ex = examples[0]
            logger.info(f"\n  [{regime.upper()}] Date: {ex['date']}")
            logger.info(f"    Delta = {ex['delta']:.4f}")
            logger.info(f"    d_LLM = {ex['analyst_decision']}")
            logger.info(f"    pi_RL = [{', '.join(f'{p:.4f}' for p in ex['pi_rl'])}]")
            logger.info(
                f"    RL argmax: {ex['rl_argmax']}, C-Gate action: {ex['cgate_action']}, "
                f"Guardian action: {ex['guardian_action']}, scale: {ex['position_scale']:.2f}"
            )
        else:
            logger.info(f"\n  [{regime.upper()}] No examples found.")

    # Warn if too much conflict
    conflict_pct = regimes.get("conflict", 0) / n if n > 0 else 0
    if conflict_pct > 0.5:
        logger.warning(
            f"WARNING: {conflict_pct:.0%} of timesteps are in CONFLICT regime under "
            f"benign conditions. This is expected with Llama 3.1 8B (90% hold bias) "
            f"and a near-uniform Executor policy. Claude Sonnet in final runs will "
            f"produce more varied signals."
        )

    return {"statistics": stats, "results": results}


def main():
    parser = argparse.ArgumentParser(description="C-Gate integration test")
    parser.add_argument(
        "--seed",
        type=int,
        default=7777,
        help="Executor seed to use (default: 7777, primary seed)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Data split (default: test)",
    )
    parser.add_argument("--tau-low", type=float, default=0.1)
    parser.add_argument("--tau-high", type=float, default=0.4)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for policy distribution (default: 1.0). "
        "Values < 1.0 sharpen the distribution.",
    )
    parser.add_argument(
        "--signals-path",
        default="data/processed/precomputed_signals.json",
        help="Path to precomputed Analyst signals JSON "
        "(default: data/processed/precomputed_signals.json). "
        "Use data/processed/precomputed_signals_gpt5.json for GPT-5 signals.",
    )
    parser.add_argument(
        "--no-guardian",
        action="store_true",
        help="Disable Guardian (no position scaling, no stop-loss, no circuit breakers).",
    )
    args = parser.parse_args()

    model_dir = f"experiments/executor/frozen/seed_{args.seed}"
    if not Path(model_dir).exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.info(
            "Available seeds: "
            + ", ".join(
                p.name.replace("seed_", "")
                for p in sorted(Path("experiments/executor/frozen").glob("seed_*"))
            )
        )
        return

    output = run_cgate_integration(
        model_dir=model_dir,
        split=args.split,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        temperature=args.temperature,
        signals_path=args.signals_path,
        enable_guardian=not args.no_guardian,
    )

    # Save results
    output_dir = Path("experiments/cgate")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"integration_{args.split}_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Log to W&B
    try:
        import wandb

        wandb.init(
            project="robust-trinity",
            name=f"cgate-integration-{args.split}-s{args.seed}",
            reinit=True,
        )
        stats = output["statistics"]
        wandb.log(
            {
                "cgate/delta_mean": stats["delta_mean"],
                "cgate/delta_median": stats["delta_median"],
                "cgate/delta_std": stats["delta_std"],
                "cgate/pct_agreement": stats["pct_agreement"],
                "cgate/pct_ambiguity": stats["pct_ambiguity"],
                "cgate/pct_conflict": stats["pct_conflict"],
                "cgate/sharpe_ratio": stats["sharpe_ratio"],
                "cgate/sortino_ratio": stats["sortino_ratio"],
                "cgate/total_return": stats["total_return"],
                "cgate/max_drawdown": stats["max_drawdown"],
            }
        )
        deltas = np.array([r["delta"] for r in output["results"]])
        wandb.log({"cgate/delta_histogram": wandb.Histogram(deltas, num_bins=50)})
        wandb.finish()
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")


if __name__ == "__main__":
    main()
