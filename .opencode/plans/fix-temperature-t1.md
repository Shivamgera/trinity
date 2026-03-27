# Plan: Fix Temperature to T=1.0 and Re-run Evaluations

## Context
T=0.05 collapsed Δ distribution to bimodal {0, 1}, making τ thresholds meaningless
(τ_low=0.9999, τ_high=1.0000). Decision: switch to T=1.0 (raw policy probabilities).

## Step 1: Update temperature defaults (3 files, 4 edits)

### Edit 1: `scripts/run_adversarial.py` line 56
```
OLD: BEST_TEMPERATURE = 0.05
NEW: BEST_TEMPERATURE = 1.0
```

### Edit 2: `scripts/calibrate_and_run.py` lines 40-41
```
OLD: default=0.01,
     help="Softmax temperature (default: 0.01)",
NEW: default=1.0,
     help="Softmax temperature (default: 1.0)",
```

### Edit 3: `src/cgate/calibrate.py` line 39 (collect_val_deltas)
```
OLD: temperature: float = 0.01,
NEW: temperature: float = 1.0,
```

### Edit 4: `src/cgate/calibrate.py` line 108 (calibrate_thresholds)
```
OLD: temperature: float = 0.01,
NEW: temperature: float = 1.0,
```

## Step 2: Recalibrate thresholds

```bash
PYTHONPATH=/Users/shivamgera/projects/research1 .venv/bin/python3 scripts/calibrate_and_run.py \
  --temperature 1.0 \
  --signals-path data/processed/precomputed_signals_gpt5.json \
  --low-pct 30 --high-pct 80 \
  --split both
```

**Verify:** New τ values should have reasonable spread (not degenerate 0.9999/1.0000).

## Step 3: Update locked tau values in `scripts/run_adversarial.py`

Take new τ_low and τ_high from Step 2 output and update lines 57-58:
```
BEST_TAU_LOW = <new_value>
BEST_TAU_HIGH = <new_value>
```

## Step 4: Re-run adversarial evaluations

```bash
PYTHONPATH=/Users/shivamgera/projects/research1 .venv/bin/python3 scripts/run_adversarial.py
```

## Step 5: Log results as Section 20 in `docs/experiment-log.md`

Document: temperature rationale, new τ values, Trinity integration results,
adversarial results, comparison with T=0.05 results.

## What does NOT need re-running
- Baselines (Executor-Only, Analyst-Only, Trinity-no-CGate) — no C-Gate temperature
- PPO training — frozen seeds unchanged