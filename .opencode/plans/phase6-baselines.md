# Phase 6: Baselines Implementation Plan

## Overview

Create a single unified script `scripts/run_baselines.py` that implements 3 baselines (Executor-only, Analyst-only, Trinity-no-CGate). The Fused LLM-RL baseline is deferred. All baselines use the same portfolio tracking / return calculation / output format as `cgate_integration.py` for fair comparison.

## Shared Infrastructure

All baselines reuse:
- Same `make_trading_env(split=..., random_start=False)` for env creation
- Same portfolio tracking: $100k initial, compound daily returns, peak tracking for MaxDD
- Same metrics: `compute_sharpe_ratio()`, `compute_max_drawdown()`
- Same output format: `{"statistics": {...}, "results": [...]}`
- Same seeds: [123, 789, 2048, 7777] from `selection.json`
- Same splits: val + test
- Same GPT-5 signals: `data/processed/precomputed_signals_gpt5.json`
- Results saved to `experiments/baselines/`

## Baseline 1: Executor-Only

**Decision logic:** At each step, compute `argmax(π_RL)` (deterministic action from PPO policy) and execute directly. No C-Gate, no Guardian, no Analyst signal.

**Key details:**
- Load frozen PPO model + VecNormalize for each seed
- Use raw logits (no temperature scaling — temperature is a C-Gate concept)
- Full position (scale=1.0), no stop-loss, no circuit breakers
- Still track portfolio value, peak, drawdown for fair MaxDD comparison

**What this tests:** Performance of the RL agent alone — the "control" for measuring C-Gate's value-add.

## Baseline 2: Analyst-Only

**Decision logic:** At each step, look up the precomputed GPT-5 signal for that date. Map: buy→long (action=1), sell→short (action=2), hold→flat (action=0). Execute at full position.

**Key details:**
- No PPO model needed — just step through env with signal-derived actions
- Missing signal dates → default to flat (action=0)
- Full position (scale=1.0), no stop-loss, no circuit breakers
- No seed variation — same signals for all runs. BUT: env is deterministic with random_start=False, so there's effectively only 1 trajectory. We run it once per split, not per-seed.

**What this tests:** Can the LLM alone generate alpha? If yes, the C-Gate's value is in risk modulation, not signal quality. If no, the C-Gate provides safety when both agents are weak.

## Baseline 3: Trinity-no-CGate

**Decision logic:** At each step:
1. Compute `argmax(π_RL)` from PPO policy
2. Look up `d_LLM` from precomputed signals
3. If `argmax(π_RL) == d_LLM` (mapped to action index): full conviction → execute argmax at position scale 1.0
4. If they disagree: execute `argmax(π_RL)` at 50% position scale
5. No Δ computation, no temperature, no thresholds

**Key details:**
- Needs both PPO model and precomputed signals
- Seed-dependent (different PPO policies → different agreement patterns)
- No Guardian (no stop-loss, no circuit breakers) — pure agree/disagree heuristic
- Missing signals → treat as disagree, execute argmax at 50%

**What this tests:** Whether the C-Gate's probabilistic Δ metric adds value over a simple binary agree/disagree check. If Trinity-no-CGate performs similarly to full Trinity, the Δ computation is unnecessary overhead.

## CLI Design

```
python scripts/run_baselines.py --baseline executor-only --split test
python scripts/run_baselines.py --baseline analyst-only --split both
python scripts/run_baselines.py --baseline trinity-no-cgate --split test --seed 789
python scripts/run_baselines.py --baseline all --split both
```

Arguments:
- `--baseline`: {executor-only, analyst-only, trinity-no-cgate, all}
- `--split`: {val, test, both} (default: both)
- `--seed`: specific seed or None for all seeds
- `--signals-path`: path to precomputed signals (default: gpt5)
- `--output-dir`: where to save results (default: experiments/baselines/)

## Output Structure

```
experiments/baselines/
  executor_only_val_seed123.json
  executor_only_val_seed789.json
  executor_only_val_seed2048.json
  executor_only_val_seed7777.json
  executor_only_test_seed123.json
  ...
  analyst_only_val.json           # No seed variation
  analyst_only_test.json
  trinity_no_cgate_val_seed123.json
  ...
```

Each JSON has the same structure as cgate_integration output:
```json
{
  "statistics": {
    "baseline": "executor-only",
    "model_dir": "...",
    "split": "test",
    "n_timesteps": 116,
    "sharpe_ratio": ...,
    "total_return": ...,
    "max_drawdown": ...,
    "final_portfolio_value": ...,
    // baseline-specific fields
  },
  "results": [
    {
      "date": "...",
      "action": ...,
      "position": ...,
      "effective_position": ...,
      "price": ...,
      "step_return": ...,
      "portfolio_value": ...,
      // baseline-specific fields
    }
  ]
}
```

## Implementation Order

1. Write the shared portfolio tracking / return calculation as a helper function
2. Implement Executor-only (simplest — just argmax + step)
3. Implement Analyst-only (signal lookup + step)
4. Implement Trinity-no-CGate (argmax + signal + agree/disagree logic)
5. Add CLI with `--baseline all` to run everything
6. Run all baselines on val + test, all seeds
7. Generate comparison table (all baselines + Trinity configs)

## Testing

- Run each baseline individually and verify output structure
- Sanity check: Executor-only results should roughly match the PPO-only numbers from training evaluation (not exact match due to different return calc)
- Analyst-only: verify action distribution matches GPT-5 signal distribution (60% hold, 22% buy, 18% sell)
- Trinity-no-CGate: verify agreement rate is reasonable (should be < 50% given the different modalities)

## Deferred: Fused LLM-RL Baseline

Requires:
1. Modify `TradingEnv` observation space to include +1 dim for LLM decision encoding
2. Create data pipeline to inject LLM signals during training
3. Retrain 4 seeds with augmented obs
4. Freeze and evaluate through same framework

This will be done after the 3 simpler baselines are working.
