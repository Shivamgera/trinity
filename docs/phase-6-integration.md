# Phase 6: Trinity Integration & Baselines

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Timeline:** Weeks 8–9 (~10–14 hours)
**Project Root:** /Users/shivamgera/projects/research1

---

## P6-T1: Implement Full Trinity Pipeline

**Estimated time:** ~3 hours
**Dependencies:** Phases 0–5 complete (frozen PPO model, pre-computed LLM signals, C-Gate, Guardian all exist)

### Context

The Robust Trinity architecture consists of three agents — Analyst (LLM-based), Executor (PPO-based), and Guardian (rule-based) — orchestrated through a Consistency Gate (C-Gate). All individual components have been implemented and tested in isolation during Phases 0–5:

- **Executor:** A frozen PPO model at `experiments/executor/best_model/model.zip`. Policy distribution extraction is available via `src/executor/policy.py` which exposes `get_policy_distribution(model, obs) → np.array` returning a 3-element probability vector over {flat=0, long=1, short=2}.
- **Analyst:** Pre-computed LLM signals stored at `data/processed/precomputed_signals.json`. Each entry maps a headline (or timestamp key) to `(decision, reasoning)`. The Analyst outputs (decision, reasoning). The C-Gate computes Δ = 1 - π_RL(d_LLM) directly from the Analyst's discrete decision and the Executor's policy distribution.
- **C-Gate:** `src/cgate/gate.py` exposes `ConsistencyGate.evaluate(d_LLM, π_RL, reasoning) → CGateResult`. `CGateResult` contains: `delta` (Δ = 1 - π_RL(d_LLM), bounded [0,1]), `regime` (one of "agreement", "ambiguity", "conflict"), `merged_action` (int), `reasoning_forwarded` (bool).
- **Guardian:** `src/guardian/` implements two stages. Stage 1: hard constraints (position limits, drawdown circuit breaker, exposure caps) checked pre-C-Gate or on the merged action. Stage 2: Δ-aware adaptive rules applied post-C-Gate (e.g., scale down position in conflict regime). The `Guardian` class composes both stages via `Guardian.process()`.

No component currently orchestrates the full per-timestep flow. This task creates that orchestration layer.

### Objective

Create the `TrinityPipeline` class that chains all components into a single `step()` call, producing a fully auditable result for each timestep.

### Detailed Instructions

1. **Create the file** `src/trinity/pipeline.py`. Ensure `src/trinity/__init__.py` exists (create it if not).

2. **Define data structures** at the top of the file using Pydantic v2 `BaseModel` or `dataclasses`:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PortfolioState:
    position: int          # current position: -1 (short), 0 (flat), 1 (long)
    cash: float            # available cash
    pnl: float             # cumulative PnL
    peak_value: float      # peak portfolio value (for drawdown tracking)
    current_value: float   # current portfolio value
    time_step: int         # current timestep in episode

@dataclass
class TrinityResult:
    final_action: int              # 0=flat, 1=long, 2=short (after all stages)
    position_scale: float          # scaling factor from Guardian Stage 2 (1.0 = full, <1.0 = reduced)
    executor_probs: np.ndarray     # π_RL — raw 3-element distribution from Executor
    analyst_decision: str          # d_LLM — Analyst's discrete decision
    delta: float                   # Δ = 1 - π_RL(d_LLM) from C-Gate
    regime: str                    # "agreement", "ambiguity", or "conflict"
    guardian_violations: list       # list of violated hard constraints (empty if none)
    reasoning_logged: str | None   # Analyst reasoning (forwarded only if regime warrants it)
```

3. **Implement `TrinityPipeline`:**

```python
class TrinityPipeline:
    def __init__(
        self,
        executor_model_path: str,
        precomputed_signals_path: str,
        guardian_config: dict,
        cgate_config: dict,
    ):
        # Load frozen PPO model using stable_baselines3
        # Load precomputed signals JSON into a dict
        # Instantiate Guardian with guardian_config
        # Instantiate ConsistencyGate with cgate_config
        ...
```

4. **Implement `step()` method** with the following flow, in strict order:

   **Step 1 — Executor inference:**
   - Call `get_policy_distribution(self.model, numeric_obs)` from `src/executor/policy.py`.
   - `numeric_obs` is a numpy array of shape `(423,)`: 30 timesteps × 14 features = 420 market features + 3 portfolio state features.
   - Store result as `pi_rl` (np.array of shape `(3,)`).
   - **Edge case:** If `numeric_obs` is all zeros (e.g., market closed), log a warning and set `pi_rl = np.array([0.34, 0.33, 0.33])` (neutral).

   **Step 2 — Analyst lookup:**
   - If `headline` is not None, look up the headline (or its timestamp key) in `self.precomputed_signals`.
    - If found, extract `(decision, reasoning)`. Pass `d_llm` to the C-Gate along with π_RL.
   - If `headline` is None or not found in precomputed signals, set `d_llm = "hold"` (neutral default) and `reasoning = None`.

   **Step 3 — C-Gate evaluation:**
   - Call `self.cgate.evaluate(d_llm, pi_rl, reasoning)` → `cgate_result`.
   - Extract `delta`, `regime`, `merged_action`, `reasoning_forwarded`.

   **Step 4 — Guardian Stage 1 (hard constraints):**
   - Call the Guardian's Stage 1 check on `cgate_result.merged_action` and `portfolio_state`.
   - If a hard constraint is violated (e.g., position limit exceeded, drawdown circuit breaker triggered), the Guardian overrides the action (typically to `flat=0`).
   - Collect all violations into a list.

   **Step 5 — Guardian Stage 2 (Δ-aware adaptive):**
   - Pass the (possibly overridden) action, `regime`, `portfolio_state`, and `reasoning` through Guardian Stage 2.
   - Stage 2 may scale the position (e.g., in conflict regime, reduce position size to 50%).
   - Extract `final_action` and `position_scale`.

   **Step 6 — Assemble and return TrinityResult:**
   - Populate all fields. For `reasoning_logged`, include analyst reasoning only if `cgate_result.reasoning_forwarded` is True.

5. **Handle additional edge cases explicitly:**
   - If `portfolio_state` indicates the portfolio is at its maximum position and the action would increase exposure further, Guardian Stage 1 must catch this.
   - All numpy arrays in the result should be copies (not references) to prevent mutation.
   - Add type hints to all methods.

6. **Write an integration test** in `tests/test_trinity_pipeline.py`:
   - Load real test data (a slice of the processed dataset).
   - Initialize `TrinityPipeline` with actual model and signal paths.
   - Run 50 timesteps in a loop, feeding sequential observations.
   - Assert for each timestep:
     - `final_action` is in {0, 1, 2}
      - `executor_probs` sums to ~1.0 (within tolerance 1e-5)
      - `analyst_decision` is one of {"hold", "buy", "sell"}
     - `delta` is in [0, 1]
     - `regime` is one of the three valid strings
     - `guardian_violations` is a list
     - `position_scale` is in (0, 1]
   - Assert no exceptions are raised across all 50 steps.
   - Include a sub-test where all headlines are None (pure numeric mode) — should still work.
   - Include a sub-test with a contrived all-zeros observation — should produce neutral behavior.

### Acceptance Criteria

- [ ] `src/trinity/pipeline.py` exists and contains `TrinityPipeline` and `TrinityResult`
- [ ] `TrinityPipeline.step()` executes the full 6-step flow without error on real data
- [ ] Integration test passes: 50 timesteps, all fields correctly populated
- [ ] Edge cases handled: missing headlines → neutral, all-zero obs → neutral/warning, portfolio at limits → Guardian override
- [ ] No component is modified — pipeline only orchestrates existing components
- [ ] Code has type hints and docstrings on all public methods

### Files to Create/Modify

- **Create:** `src/trinity/__init__.py` (if not exists)
- **Create:** `src/trinity/pipeline.py`
- **Create:** `tests/test_trinity_pipeline.py`

### Dependencies

- Phase 5 complete (all individual components tested and working)

### Human Checkpoint

Before proceeding to P6-T2, verify:
1. Run the integration test and confirm it passes.
2. Inspect one TrinityResult manually — does the audit trail make sense? Does the regime match what you'd expect given the agent distributions?
3. Confirm that the Guardian actually overrides actions when it should (temporarily set a very low drawdown limit and verify it triggers).

---

## P6-T2: Implement All 4 Baselines

**Estimated time:** ~3 hours
**Dependencies:** P6-T1 complete (TrinityPipeline exists and works)

### Context

The thesis compares the full Trinity system against four baselines to demonstrate that (a) multi-agent orchestration adds value, (b) the C-Gate is critical for robustness, and (c) channel independence matters. Each baseline deliberately removes or alters a component:

| Baseline | Analyst | Executor | C-Gate | Guardian |
|---|---|---|---|---|
| 1. Executor-only | No | Yes | No | Stage 1 only |
| 2. Analyst-only | Yes | No | No | Stage 1 only |
| 3. Trinity-no-CGate | Yes | Yes | No (agree/disagree) | Stage 1 only |
| 4. Fused-LLM-RL | Embedded | Modified | No | Stage 1 only |

All baselines share the same `step(numeric_obs, headline, portfolio_state) → TrinityResult` interface so they can be swapped interchangeably with the full Trinity pipeline in the backtesting framework.

**Existing components available:**
- `src/executor/policy.py`: `get_policy_distribution(model, obs)` — returns π_RL
- `src/cgate/divergence.py`: `compute_delta(d_llm, pi_rl)` — returns Δ = 1 - π_RL(d_LLM)
- `data/processed/precomputed_signals.json` — pre-computed Analyst signals
- `experiments/executor/best_model/model.zip` — frozen PPO model (obs_dim=423)
- Guardian Stage 1 hard constraints (from `src/guardian/`)
- `TrinityResult` dataclass (from P6-T1)

### Objective

Implement four baseline systems with identical interfaces to `TrinityPipeline`, plus a training script for the Fused-LLM-RL variant's modified PPO model.

### Detailed Instructions

1. **Create `src/trinity/baselines.py`.**

2. **Baseline 1 — `ExecutorOnly`:**

```python
class ExecutorOnly:
    def __init__(self, executor_model_path: str, guardian_config: dict):
        # Load frozen PPO model
        # Instantiate Guardian (Stage 1 only)

    def step(self, numeric_obs: np.ndarray, headline: str | None, portfolio_state: PortfolioState) -> TrinityResult:
        # 1. Get π_RL from executor
        pi_rl = get_policy_distribution(self.model, numeric_obs)
        # 2. Take argmax action
        raw_action = int(np.argmax(pi_rl))
        # 3. Guardian Stage 1 check
        action, violations = self.guardian.check_stage1(raw_action, portfolio_state)
        # 4. Return TrinityResult with:
        #    - analyst_decision = None or "n/a"
        #    - delta = NaN or 0.0 (no C-Gate)
        #    - regime = "n/a"
        #    - position_scale = 1.0 (no Stage 2 scaling)
```

   - The `headline` parameter is accepted but **ignored** (this baseline uses no text).

3. **Baseline 2 — `AnalystOnly`:**

```python
class AnalystOnly:
    def __init__(self, precomputed_signals_path: str, guardian_config: dict):
        # Load precomputed signals
        # Instantiate Guardian (Stage 1 only)

    def step(self, numeric_obs: np.ndarray, headline: str | None, portfolio_state: PortfolioState) -> TrinityResult:
        # 1. Look up headline in precomputed signals
        # 2. If found: use d_LLM directly as the action (map hold→0, buy→1, sell→2)
        # 3. If no headline: default to flat (action=0) — this baseline can't act without text
        # 4. Guardian Stage 1 check
        # 5. Return TrinityResult with:
        #    - executor_probs = None or np.zeros(3)
        #    - delta = NaN or 0.0
        #    - regime = "n/a"
```

   - The `numeric_obs` parameter is accepted but **ignored**.
   - When no headline is available, the Analyst-only system defaults to `flat` (action=0). This is a known weakness that the thesis should discuss — an Analyst-only system is blind during periods without news.

4. **Baseline 3 — `TrinityNoCGate`:**

```python
class TrinityNoCGate:
    def __init__(self, executor_model_path: str, precomputed_signals_path: str, guardian_config: dict):
        # Load both models
        # Instantiate Guardian (Stage 1 only)

    def step(self, numeric_obs: np.ndarray, headline: str | None, portfolio_state: PortfolioState) -> TrinityResult:
        # Trinity-no-CGate baseline:
        # 1. Get π_RL from Executor (same as full Trinity)
        # 2. Get d_LLM from Analyst signal (same as full Trinity)
        # 3. Simple agreement check:
        #    - If argmax(π_RL) == d_LLM → execute argmax(π_RL) at full position size
        #    - If argmax(π_RL) != d_LLM → execute argmax(π_RL) at 50% position size
        #    - No Δ computation, no regime classification
        # 4. Guardian Stage 1 hard constraints still apply
        #
        # This baseline tests the hypothesis that the continuous Δ-based regime
        # arbitration adds value beyond a simple binary agree/disagree check.
        # Still log the Δ that WOULD have been computed (for comparison plots).
        # 6. Return TrinityResult with:
        #    - delta = compute Δ anyway (for logging), but it's not USED for decisions
        #    - regime = "n/a" (no regime-based arbitration)
        #    - position_scale = 1.0 or 0.5 depending on agreement
```

   - This baseline tests the hypothesis that continuous Δ-based regime arbitration adds value beyond a simple binary agree/disagree check.
   - Still compute and log Δ so we can compare distributions with the full Trinity.

5. **Baseline 4 — `FusedLLMRL`:**

   This is the most complex baseline. It tests the **channel independence hypothesis** by deliberately violating it: the LLM signal is injected directly into the Executor's observation space.

   **Step 5a — Create modified environment:**
   - Create `src/executor/fused_env.py` (or modify the existing `TradingEnv` to accept a `fused=True` flag).
   - The observation space changes from `(423,)` to `(424,)`:
       - First 420 dims: same 30×14 z-normalized market features
       - Next 3 dims: same portfolio state (position, PnL, time_fraction)
       - Dim 423: `decision_encoded` — float encoding of Analyst's decision: hold=0.0, buy=1.0, sell=2.0
    - During training, for each timestep, look up the corresponding headline's precomputed signal and append `(decision_encoded,)` to the observation.
    - If no headline exists for a timestep, use `0.0` — encoded as flat/hold.

   **Step 5b — Create fused training script:**
   - Create `scripts/train_fused_executor.py`.
   - This is a variant of the original PPO training script (from Phase 3) with:
     - Modified environment using the fused observation space
     - Same hyperparameters as the original PPO training
     - Save model to `experiments/executor/fused_model/model.zip`
   - Training should take roughly the same time as the original Executor training.

   **Step 5c — Implement the baseline class:**

```python
class FusedLLMRL:
    def __init__(self, fused_model_path: str, precomputed_signals_path: str, guardian_config: dict):
        # Load the FUSED PPO model (obs_dim=424)
        # Load precomputed signals
        # Instantiate Guardian (Stage 1 only)

    def step(self, numeric_obs: np.ndarray, headline: str | None, portfolio_state: PortfolioState) -> TrinityResult:
        # 1. Look up headline → decision_encoded. Default: 0.0
        # 2. Concatenate: fused_obs = np.concatenate([numeric_obs, [decision_encoded]])
        #    Assert fused_obs.shape == (424,)
        # 3. Get π_fused from fused model
        # 4. Argmax → action
        # 5. Guardian Stage 1 check
        # 6. Return TrinityResult
```

   **IMPORTANT NOTE:** Training the fused model may take 1–2 hours depending on hardware. The human should decide whether to train it now or defer. If deferred, create a placeholder `FusedLLMRL` class that raises `NotImplementedError("Fused model not yet trained. Run scripts/train_fused_executor.py first.")`.

6. **Write tests** in `tests/test_baselines.py`:
   - For each of the 4 baselines (or 3, if Fused is deferred):
     - Initialize with real model/signal paths
     - Run 20 timesteps
     - Assert `TrinityResult` fields are correctly populated
      - Assert the correct components are used / ignored (e.g., ExecutorOnly should have `analyst_decision` as None or "n/a")
   - Comparative test: run all systems on the same 20 timesteps, verify they produce different actions (if they all produce identical actions, something is wrong)

### Acceptance Criteria

- [ ] `src/trinity/baselines.py` contains `ExecutorOnly`, `AnalystOnly`, `TrinityNoCGate`, `FusedLLMRL`
- [ ] All 4 classes implement the same `step()` interface as `TrinityPipeline`
- [ ] `ExecutorOnly` ignores headlines; `AnalystOnly` ignores numeric_obs
- [ ] `TrinityNoCGate` uses simple agreement check without regime-based decisions
- [ ] `FusedLLMRL` has either a working fused model or a clear placeholder with training script ready
- [ ] `scripts/train_fused_executor.py` exists and is runnable
- [ ] `src/executor/fused_env.py` exists with modified observation space (424 dims)
- [ ] Tests pass for all implemented baselines
- [ ] Each baseline's `TrinityResult` clearly indicates which components are active/inactive

### Files to Create/Modify

- **Create:** `src/trinity/baselines.py`
- **Create:** `src/executor/fused_env.py`
- **Create:** `scripts/train_fused_executor.py`
- **Create:** `tests/test_baselines.py`

### Dependencies

- P6-T1 (TrinityPipeline and TrinityResult must exist)

### Human Checkpoint

Before proceeding to P6-T3, verify:
1. Run all baseline tests and confirm they pass.
2. **Decision required:** Train the fused model now or defer? If deferring, ensure the placeholder raises a clear error.
3. Manually compare: run each baseline on the same few timesteps and verify the outputs make intuitive sense. For example, on a timestep with a strongly bullish headline, the AnalystOnly should lean toward `long`, while ExecutorOnly might differ.
4. Confirm that `TrinityNoCGate` produces different actions than the full Trinity on at least some timesteps (demonstrating that C-Gate arbitration matters).

---

## P6-T3: Backtesting Framework and Benign Evaluation

**Estimated time:** ~3 hours
**Dependencies:** P6-T1 and P6-T2 complete (TrinityPipeline and all baselines available)

### Context

With the full Trinity pipeline and all 4 baselines implemented, we need a backtesting framework that can:
1. Run any pipeline over historical data while tracking portfolio state.
2. Compute all financial metrics (PnL, Sharpe, Max Drawdown, VaR, Expected Shortfall).
3. Run the same data through multiple pipelines for apples-to-apples comparison.
4. Produce a comparison table for the benign (no-attack) condition.

**Existing components available:**
- `src/evaluation/financial.py` with: `compute_pnl()`, `compute_sharpe()`, `compute_max_drawdown()`, `compute_var()`, `compute_expected_shortfall()`
- `TrinityPipeline` (P6-T1) and all baselines (P6-T2) with shared `step()` interface
- `data/processed/` contains processed market data (numeric features) and `precomputed_signals.json` (Analyst signals)
- W&B integration is already configured in the project

**Data layout assumed:**
- Numeric features: `data/processed/features.parquet` or similar — DataFrame with datetime index, columns for 14 z-normalized features per timestep. The Executor expects a flattened window of 30 timesteps × 14 features = 420 dims + 3 portfolio state dims = 423 total.
- Headlines: `data/processed/precomputed_signals.json` — keyed by date or timestamp.
- Test period: July 2024 – December 2024 (out-of-sample for the PPO model which was trained on earlier data).

### Objective

Build a backtesting engine, run benign evaluation across all 5 systems with 5 seeds, and produce a comparison table establishing baseline performance.

### Detailed Instructions

1. **Create `src/trinity/backtest.py`.**

2. **Implement data loading utility:**

```python
class DataLoader:
    def __init__(self, features_path: str, signals_path: str):
        # Load numeric features (DataFrame with datetime index)
        # Load precomputed signals (dict keyed by date string)

    def get_window(self, idx: int, window_size: int = 30) -> np.ndarray:
        # Return flattened feature window: shape (420,)
        # Handles boundary: if idx < window_size, pad with zeros

    def get_headline(self, date: str) -> str | None:
        # Look up headline for this date. Return None if no headline.

    def get_price(self, idx: int) -> float:
        # Return the closing price at this index (for PnL computation)

    def __len__(self) -> int:
        # Number of tradeable timesteps
```

   - Handle date alignment carefully: numeric data is daily, headlines may not exist for every day.

3. **Implement `Backtester`:**

```python
@dataclass
class BacktestResult:
    daily_pnl: list[float]          # PnL for each day
    cumulative_pnl: list[float]     # running cumulative PnL
    positions: list[int]            # position at each timestep
    actions: list[int]              # action taken at each timestep
    deltas: list[float]             # Δ = 1 - π_RL(d_LLM) at each timestep (NaN for baselines without C-Gate)
    regimes: list[str]              # regime at each timestep
    portfolio_values: list[float]   # portfolio value at each timestep
    dates: list[str]                # corresponding dates

    # Computed metrics (populated after run)
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    expected_shortfall_95: float = 0.0

class Backtester:
    def __init__(
        self,
        pipeline,               # TrinityPipeline or any baseline (duck-typed via step())
        data_loader: DataLoader,
        config: dict,           # initial_cash, transaction_cost, etc.
    ):
        self.pipeline = pipeline
        self.data = data_loader
        self.initial_cash = config.get("initial_cash", 100_000)
        self.transaction_cost = config.get("transaction_cost", 0.001)  # 10 bps

    def run(self, start_date: str, end_date: str, seed: int) -> BacktestResult:
        np.random.seed(seed)
        # Initialize PortfolioState
        portfolio = PortfolioState(
            position=0, cash=self.initial_cash, pnl=0.0,
            peak_value=self.initial_cash, current_value=self.initial_cash, time_step=0
        )
        # Filter data to [start_date, end_date]
        # For each timestep:
        #   1. Construct numeric_obs: feature window (420) + portfolio state (3) = (423,)
        #      portfolio state features = [position_encoded, normalized_pnl, time_fraction]
        #   2. Get headline for this date (may be None)
        #   3. Call pipeline.step(numeric_obs, headline, portfolio) → TrinityResult
        #   4. Execute the action:
        #      - If action changes position, compute transaction cost
        #      - Update cash based on price change and position
        #      - Update portfolio value, PnL, peak value
        #   5. Record all fields in result lists
        # After loop: compute all financial metrics using src/evaluation/financial.py
        # Return BacktestResult
```

   **Portfolio simulation details:**
   - Position is the target: action=0 → flat, action=1 → long 1 unit, action=2 → short 1 unit.
   - If current position matches action, no trade (no cost).
   - If position changes, apply transaction cost: `cost = abs(position_change) × price × transaction_cost_rate`.
   - Daily PnL = position × (price_today - price_yesterday) - transaction_costs.
   - Track cumulative PnL, portfolio value (initial_cash + cumulative_pnl), and update peak for drawdown.

4. **Implement `run_comparison()`:**

```python
import pandas as pd

def run_comparison(
    pipelines: dict[str, object],  # {"Trinity": trinity_pipeline, "Executor-only": exec_only, ...}
    data_loader: DataLoader,
    backtest_config: dict,
    start_date: str,
    end_date: str,
    seeds: list[int],
    wandb_project: str | None = None,
) -> pd.DataFrame:
    """
    Run all pipelines on the same data with the same seeds.
    Returns DataFrame: rows = pipeline names, columns = metric (mean ± std).
    """
    all_results = {}  # {pipeline_name: [BacktestResult for each seed]}
    for name, pipeline in pipelines.items():
        seed_results = []
        for seed in seeds:
            bt = Backtester(pipeline, data_loader, backtest_config)
            result = bt.run(start_date, end_date, seed)
            seed_results.append(result)
            # Log to W&B if configured
            if wandb_project:
                import wandb
                wandb.init(project=wandb_project, name=f"{name}_seed{seed}",
                           tags=[name, "benign", f"seed_{seed}"], reinit=True)
                wandb.log({
                    "total_pnl": result.total_pnl,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "var_95": result.var_95,
                    "es_95": result.expected_shortfall_95,
                })
                wandb.finish()
        all_results[name] = seed_results

    # Build comparison DataFrame
    rows = []
    for name, results in all_results.items():
        metrics = {
            "PnL": [r.total_pnl for r in results],
            "Sharpe": [r.sharpe_ratio for r in results],
            "MDD": [r.max_drawdown for r in results],
            "VaR(95%)": [r.var_95 for r in results],
            "ES(95%)": [r.expected_shortfall_95 for r in results],
        }
        row = {"System": name}
        for metric_name, values in metrics.items():
            row[f"{metric_name}_mean"] = np.mean(values)
            row[f"{metric_name}_std"] = np.std(values)
            row[metric_name] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("System")
```

5. **Create the benign evaluation script** `scripts/run_benign_evaluation.py`:

```python
"""
Run benign (no-attack) evaluation of Trinity + all baselines.
Usage: python scripts/run_benign_evaluation.py
"""
# 1. Instantiate all 5 pipelines (Trinity + 4 baselines)
#    Use paths: experiments/executor/best_model/model.zip, data/processed/precomputed_signals.json
#    For FusedLLMRL: use experiments/executor/fused_model/model.zip (or skip if not trained)
# 2. Create DataLoader with test period data
# 3. Run run_comparison() with:
#    - start_date = "2024-07-01"
#    - end_date = "2024-12-31"
#    - seeds = [42, 123, 456, 789, 1024]
#    - wandb_project = "robust-trinity-benign"
# 4. Print comparison table
# 5. Save table to experiments/results/benign_comparison.csv
# 6. Save raw results to experiments/results/benign_raw.json (for later statistical analysis)
```

6. **Run the benign evaluation:**
   - Execute the script.
   - Verify all 5 systems complete without errors across all 5 seeds (25 total runs).
   - Inspect the comparison table.

7. **Write tests** in `tests/test_backtest.py`:
   - Test `Backtester` on a small synthetic dataset (10 timesteps, known prices):
     - Create a fake pipeline that always returns action=1 (long).
     - Verify PnL matches manual calculation.
   - Test `DataLoader` date alignment: verify correct handling of missing headlines.
   - Test portfolio state updates: position transitions, transaction costs, drawdown tracking.
   - Test that `run_comparison` returns a DataFrame with correct shape and columns.

### Acceptance Criteria

- [ ] `src/trinity/backtest.py` contains `DataLoader`, `Backtester`, `BacktestResult`, and `run_comparison()`
- [ ] `scripts/run_benign_evaluation.py` exists and runs successfully
- [ ] Benign evaluation completed: 5 systems × 5 seeds = 25 runs, no failures
- [ ] Comparison table saved to `experiments/results/benign_comparison.csv`
- [ ] All financial metrics (PnL, Sharpe, MDD, VaR, ES) are computed and reasonable
- [ ] W&B dashboard shows all 25 runs with correct tags
- [ ] Tests pass for backtester, data loader, and portfolio simulation
- [ ] Trinity performance is competitive with (not necessarily better than) the best single-agent baseline under benign conditions

### Files to Create/Modify

- **Create:** `src/trinity/backtest.py`
- **Create:** `scripts/run_benign_evaluation.py`
- **Create:** `tests/test_backtest.py`
- **Create:** `experiments/results/` directory (if not exists)
- **Create:** `experiments/results/benign_comparison.csv` (output)
- **Create:** `experiments/results/benign_raw.json` (output)

### Dependencies

- P6-T1 (TrinityPipeline)
- P6-T2 (all baselines)

### Human Checkpoint

Before proceeding to Phase 7, review the benign comparison table carefully:
1. **Sanity check:** Are all Sharpe ratios in a reasonable range (e.g., -1 to 3 for a 6-month period)? Is PnL non-degenerate (not always 0)?
2. **Key question:** Is Trinity competitive with Executor-only? If Trinity is significantly WORSE than Executor-only under benign conditions, there may be a bug in the C-Gate or Guardian that's being too conservative. Investigate before proceeding.
3. **Analyst-only baseline:** This should likely underperform since it can only act when headlines are available. Confirm this and note the number of "no headline" timesteps.
4. **Reproducibility:** Verify that running the same seed twice produces identical results.
5. **Decision point:** If FusedLLMRL is not yet trained, decide whether to train it now (~1-2 hours) or proceed to Phase 7 and return to it later.
