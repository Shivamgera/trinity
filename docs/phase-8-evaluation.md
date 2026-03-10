# Phase 8: Full Evaluation & Results

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Timeline:** Weeks 10–12 (~20–28 hours)
**Project Root:** /Users/shivamgera/projects/research1

---

## P8-T1: Build Experiment Runner and Metrics Infrastructure

**Estimated time:** ~3 hours
**Dependencies:** Phase 6 (Trinity pipeline, baselines, backtester) and Phase 7 (all attacks) complete.

### Context

Phases 6 and 7 produced all the components needed for evaluation:
- 5 trading systems: Trinity (`src/trinity/pipeline.py`), Executor-only, Analyst-only, Trinity-no-CGate, Fused-LLM-RL (all in `src/trinity/baselines.py`)
- 12 attack configurations in `configs/adversarial/*.yaml` plus a benign (no-attack) condition = 13 conditions
- Backtesting framework in `src/trinity/backtest.py` with `Backtester.run()` returning `BacktestResult`
- Financial metrics in `src/evaluation/financial.py`: `compute_pnl()`, `compute_sharpe()`, `compute_max_drawdown()`, `compute_var()`, `compute_expected_shortfall()`
- Attack factory in `src/adversarial/__init__.py`: `load_attack(config_path, feature_stds, seed)`

The full experiment matrix is: 5 systems × 13 conditions × 5 seeds = 325 runs. Each run feeds one system through the backtester on the test period (Jul–Dec 2024) under one attack condition with one random seed.

Additionally, we need robustness-specific metrics beyond the financial ones:
- **Performance degradation:** How much does each metric worsen under attack vs. benign?
- **Action stability (flip rate):** What fraction of actions change under attack compared to benign?
- **Regime stability:** How stable are C-Gate regime assignments over time?

### Objective

Build the experiment runner that orchestrates all 325 runs, computes all metrics (financial + robustness), saves results, and logs to W&B.

### Detailed Instructions

1. **Create `src/evaluation/runner.py`.**

2. **Define experiment configuration:**

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ExperimentConfig:
    # Data
    features_path: str = "data/processed/features.parquet"
    signals_path: str = "data/processed/precomputed_signals.json"
    feature_stds_path: str = "data/processed/feature_stds.npy"

    # Models
    executor_model_path: str = "experiments/executor/best_model/model.zip"
    fused_model_path: str = "experiments/executor/fused_model/model.zip"

    # Evaluation period
    start_date: str = "2024-07-01"
    end_date: str = "2024-12-31"

    # Experiment parameters
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    attack_configs_dir: str = "configs/adversarial"

    # Portfolio simulation
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.001  # 10 bps

    # Output
    output_dir: str = "experiments/results"
    wandb_project: str = "robust-trinity"

    # Execution
    n_parallel: int = 1  # set > 1 to parallelize across seeds
```

3. **Implement `ExperimentRunner`:**

```python
class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load shared resources
        self.feature_stds = np.load(config.feature_stds_path)
        self.data_loader = DataLoader(config.features_path, config.signals_path)

        # Initialize all 5 pipelines
        self.pipelines = self._build_pipelines()

        # Load all attack configs + benign (None)
        self.attack_configs = self._load_attack_configs()

    def _build_pipelines(self) -> dict[str, object]:
        """Instantiate all 5 trading systems."""
        from src.trinity.pipeline import TrinityPipeline
        from src.trinity.baselines import ExecutorOnly, AnalystOnly, TrinityNoCGate, FusedLLMRL

        guardian_config = {}   # load from configs/guardian.yaml or use defaults
        cgate_config = {}      # load from configs/cgate.yaml or use defaults

        pipelines = {
            "Trinity": TrinityPipeline(
                self.config.executor_model_path,
                self.config.signals_path,
                guardian_config, cgate_config,
            ),
            "Executor-only": ExecutorOnly(
                self.config.executor_model_path, guardian_config,
            ),
            "Analyst-only": AnalystOnly(
                self.config.signals_path, guardian_config,
            ),
            "Trinity-no-CGate": TrinityNoCGate(
                self.config.executor_model_path,
                self.config.signals_path,
                guardian_config,
            ),
        }

        # FusedLLMRL — skip if model doesn't exist
        fused_path = Path(self.config.fused_model_path)
        if fused_path.exists():
            pipelines["Fused-LLM-RL"] = FusedLLMRL(
                str(fused_path),
                self.config.signals_path,
                guardian_config,
            )
        else:
            print(f"WARNING: Fused model not found at {fused_path}. Skipping Fused-LLM-RL baseline.")

        return pipelines

    def _load_attack_configs(self) -> dict[str, str | None]:
        """Load all attack YAML paths. 'benign' maps to None."""
        configs = {"benign": None}
        attack_dir = Path(self.config.attack_configs_dir)
        if attack_dir.exists():
            for yaml_path in sorted(attack_dir.glob("*.yaml")):
                name = yaml_path.stem  # e.g., "numeric_low"
                configs[name] = str(yaml_path)
        return configs

    def run_single(
        self,
        pipeline_name: str,
        attack_name: str,
        seed: int,
    ) -> dict:
        """
        Run a single experiment: one pipeline × one attack × one seed.
        Returns a dict with all metrics and metadata.
        """
        pipeline = self.pipelines[pipeline_name]
        attack_config_path = self.attack_configs[attack_name]

        # Create a COPY of the data loader for this run (to avoid mutation)
        # Apply attack to data if not benign
        if attack_config_path is not None:
            attack = load_attack(attack_config_path, self.feature_stds, seed)
            # Wrap the data loader to apply attack on-the-fly
            data_loader = AttackedDataLoader(self.data_loader, attack)
        else:
            data_loader = self.data_loader

        # Run backtest
        backtester = Backtester(
            pipeline, data_loader,
            {"initial_cash": self.config.initial_cash,
             "transaction_cost": self.config.transaction_cost}
        )
        result = backtester.run(self.config.start_date, self.config.end_date, seed)

        # Package results
        run_result = {
            "pipeline": pipeline_name,
            "attack": attack_name,
            "seed": seed,
            "metrics": {
                "pnl": result.total_pnl,
                "sharpe": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "var_95": result.var_95,
                "es_95": result.expected_shortfall_95,
            },
            "time_series": {
                "daily_pnl": result.daily_pnl,
                "cumulative_pnl": result.cumulative_pnl,
                "positions": result.positions,
                "actions": result.actions,
                "deltas": result.deltas,
                "regimes": result.regimes,
                "portfolio_values": result.portfolio_values,
            },
            "dates": result.dates,
        }

        # Save individual run result
        run_dir = self.output_dir / pipeline_name / attack_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / f"seed_{seed}.json", "w") as f:
            json.dump(run_result, f, default=str)  # default=str handles numpy types

        # Log to W&B
        if self.config.wandb_project:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=f"{pipeline_name}/{attack_name}/seed{seed}",
                tags=[pipeline_name, attack_name, f"seed_{seed}"],
                config={"pipeline": pipeline_name, "attack": attack_name, "seed": seed},
                reinit=True,
            )
            wandb.log(run_result["metrics"])
            # Log cumulative PnL as a line chart
            for i, (date, pnl) in enumerate(zip(result.dates, result.cumulative_pnl)):
                wandb.log({"step": i, "cumulative_pnl": pnl})
            wandb.finish()

        return run_result

    def run_all(self) -> list[dict]:
        """Run the full experiment matrix."""
        all_results = []
        total = len(self.pipelines) * len(self.attack_configs) * len(self.config.seeds)
        completed = 0

        for pipeline_name in self.pipelines:
            for attack_name in self.attack_configs:
                for seed in self.config.seeds:
                    print(f"[{completed+1}/{total}] {pipeline_name} / {attack_name} / seed={seed}")
                    try:
                        result = self.run_single(pipeline_name, attack_name, seed)
                        all_results.append(result)
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        all_results.append({
                            "pipeline": pipeline_name, "attack": attack_name,
                            "seed": seed, "error": str(e),
                        })
                    completed += 1

        # Save full results index
        with open(self.output_dir / "all_results_index.json", "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "time_series"}
                       for r in all_results], f, indent=2, default=str)

        return all_results
```

4. **Implement `AttackedDataLoader`** — a wrapper that applies attacks on-the-fly:

```python
class AttackedDataLoader:
    """Wraps a DataLoader and applies an attack to observations/headlines."""

    def __init__(self, base_loader: DataLoader, attack):
        self.base = base_loader
        self.attack = attack

    def get_window(self, idx, window_size=30):
        obs = self.base.get_window(idx, window_size)
        if hasattr(self.attack, 'perturb'):
            obs = self.attack.perturb(obs)
        return obs

    def get_headline(self, date):
        headline = self.base.get_headline(date)
        # For semantic attacks, headlines are pre-poisoned via adversarial signal files
        # The attack config points to the adversarial signals file
        # So we don't need to poison on-the-fly here
        return headline

    def get_price(self, idx):
        return self.base.get_price(idx)  # prices are never attacked

    def __len__(self):
        return len(self.base)
```

   **Design note:** Semantic attacks modify the signals file (pre-computed), not the headlines on-the-fly. The `AttackedDataLoader` handles numeric perturbation at read time. For semantic attacks, the pipeline is initialized with the adversarial signals file path instead of the benign one. The `run_single()` method must handle this — check if the attack is semantic-only and swap the signals path accordingly.

5. **Create `src/evaluation/robustness.py`:**

```python
"""Robustness-specific metrics beyond standard financial metrics."""

import numpy as np
from scipy.stats import entropy

def compute_degradation(benign_metric: float, attack_metric: float) -> float:
    """
    Compute percentage performance degradation under attack.

    For metrics where higher is better (PnL, Sharpe):
        degradation = (benign - attack) / |benign| * 100

    Returns positive values for degradation, negative for improvement.
    """
    if abs(benign_metric) < 1e-10:
        return 0.0 if abs(attack_metric) < 1e-10 else float('inf')
    return (benign_metric - attack_metric) / abs(benign_metric) * 100

def compute_degradation_directional(
    benign_metric: float,
    attack_metric: float,
    higher_is_better: bool = True,
) -> float:
    """
    Same as compute_degradation but with explicit direction.
    Returns positive for degradation (performance got worse).
    """
    if higher_is_better:
        return compute_degradation(benign_metric, attack_metric)
    else:
        # For MDD, VaR, ES: larger magnitude = worse
        return compute_degradation(-benign_metric, -attack_metric)

def compute_flip_rate(actions_benign: list[int], actions_attack: list[int]) -> float:
    """
    Fraction of timesteps where the action changed under attack.

    This measures action stability — a robust system should maintain
    consistent actions even when inputs are perturbed.

    Args:
        actions_benign: list of actions under benign conditions
        actions_attack: list of actions under attack (same seed, same data)

    Returns:
        float in [0, 1] — 0 means perfectly stable, 1 means every action flipped
    """
    assert len(actions_benign) == len(actions_attack), "Action sequences must have same length"
    if len(actions_benign) == 0:
        return 0.0
    flips = sum(a != b for a, b in zip(actions_benign, actions_attack))
    return flips / len(actions_benign)

def compute_directional_flip_rate(
    actions_benign: list[int],
    actions_attack: list[int],
) -> dict[str, float]:
    """
    Break down flips by direction.

    Returns dict with:
        "overall": total flip rate
        "to_long": fraction of non-long actions that flipped to long
        "to_short": fraction of non-short actions that flipped to short
        "to_flat": fraction of non-flat actions that flipped to flat
    """
    n = len(actions_benign)
    if n == 0:
        return {"overall": 0.0, "to_long": 0.0, "to_short": 0.0, "to_flat": 0.0}

    flips = {"overall": 0, "to_long": 0, "to_short": 0, "to_flat": 0}
    for a_b, a_a in zip(actions_benign, actions_attack):
        if a_b != a_a:
            flips["overall"] += 1
            if a_a == 1:
                flips["to_long"] += 1
            elif a_a == 2:
                flips["to_short"] += 1
            elif a_a == 0:
                flips["to_flat"] += 1

    return {k: v / n for k, v in flips.items()}

def compute_regime_stability(regimes: list[str]) -> dict[str, float]:
    """
    Compute stability metrics for C-Gate regime assignments over time.

    Returns:
        "entropy": Shannon entropy of regime distribution (lower = more concentrated)
        "transition_rate": fraction of timesteps where regime changed
        "regime_fractions": dict of fraction of time in each regime
    """
    if not regimes or all(r == "n/a" for r in regimes):
        return {"entropy": 0.0, "transition_rate": 0.0, "regime_fractions": {}}

    valid_regimes = [r for r in regimes if r != "n/a"]
    if not valid_regimes:
        return {"entropy": 0.0, "transition_rate": 0.0, "regime_fractions": {}}

    # Regime fractions
    unique_regimes = ["agreement", "ambiguity", "conflict"]
    counts = {r: 0 for r in unique_regimes}
    for r in valid_regimes:
        if r in counts:
            counts[r] += 1
    n = len(valid_regimes)
    fractions = {r: c / n for r, c in counts.items()}

    # Entropy
    probs = [fractions[r] for r in unique_regimes if fractions[r] > 0]
    regime_entropy = entropy(probs, base=2)  # bits

    # Transition rate
    transitions = sum(
        valid_regimes[i] != valid_regimes[i - 1]
        for i in range(1, len(valid_regimes))
    )
    transition_rate = transitions / max(len(valid_regimes) - 1, 1)

    return {
        "entropy": regime_entropy,
        "transition_rate": transition_rate,
        "regime_fractions": fractions,
    }
```

6. **Write tests** in `tests/test_evaluation.py`:

```python
def test_degradation_positive():
    """Positive degradation means performance worsened."""
    assert compute_degradation(1.5, 1.0) > 0   # Sharpe dropped
    assert compute_degradation(1.5, 2.0) < 0   # Sharpe improved (negative degradation)

def test_flip_rate():
    assert compute_flip_rate([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0
    assert compute_flip_rate([1, 1, 1, 1], [2, 2, 2, 2]) == 1.0
    assert compute_flip_rate([1, 1, 0, 0], [1, 2, 0, 1]) == 0.5

def test_regime_stability_all_agreement():
    result = compute_regime_stability(["agreement"] * 100)
    assert result["entropy"] == 0.0
    assert result["transition_rate"] == 0.0
    assert result["regime_fractions"]["agreement"] == 1.0

def test_regime_stability_mixed():
    regimes = ["agreement"] * 50 + ["conflict"] * 50
    result = compute_regime_stability(regimes)
    assert result["entropy"] > 0.0
    assert result["transition_rate"] < 0.05  # only one transition in the middle
```

### Acceptance Criteria

- [ ] `src/evaluation/runner.py` contains `ExperimentRunner`, `ExperimentConfig`, `AttackedDataLoader`
- [ ] `ExperimentRunner.run_all()` can orchestrate the full 325-run experiment matrix
- [ ] `ExperimentRunner.run_single()` saves per-run JSON and logs to W&B
- [ ] `src/evaluation/robustness.py` contains `compute_degradation()`, `compute_flip_rate()`, `compute_directional_flip_rate()`, `compute_regime_stability()`
- [ ] Semantic attacks correctly swap the signals path (not perturb on-the-fly)
- [ ] Failed runs are caught and logged without crashing the entire experiment
- [ ] Tests pass for all robustness metrics
- [ ] Output directory structure: `experiments/results/{pipeline}/{attack}/seed_{n}.json`

### Files to Create/Modify

- **Create:** `src/evaluation/runner.py`
- **Create:** `src/evaluation/robustness.py`
- **Create:** `tests/test_evaluation.py`
- **Create:** `experiments/results/` directory structure (created by runner)

### Dependencies

- Phase 6 complete (all pipelines and backtester)
- Phase 7 complete (all attack configs and implementations)

### Human Checkpoint

Before proceeding to P8-T2, verify:
1. Run a **mini experiment**: 2 systems x 2 attacks x 1 seed = 4 runs. Confirm outputs are saved correctly and W&B receives the logs.
2. Check timing: how long does 1 run take? Extrapolate to 325 runs. If >15 hours, consider parallelizing or reducing seeds.
3. Verify the `AttackedDataLoader` is actually applying perturbations (compare an observation before and after attack).
4. Check that W&B tags are structured for easy filtering (pipeline name, attack type, seed).

---

## P8-T2: Run Full Experiment Suite and Collect Results

**Estimated time:** ~4 hours (mostly waiting for runs to complete)
**Dependencies:** P8-T1 complete (ExperimentRunner built and verified on mini experiment)

### Context

The ExperimentRunner from P8-T1 is ready to orchestrate the full experiment matrix. The full matrix is 5 systems x 13 conditions x 5 seeds = 325 runs. At an estimated ~2 minutes per run, this takes ~11 hours sequentially.

To manage time, we prioritize runs in tiers:
- **Priority 1 (minimum viable):** Trinity + Executor-only + Analyst-only under all 13 conditions x 5 seeds = 195 runs (~6.5 hrs). This gives us the core comparison: does Trinity outperform single-agent systems under attack?
- **Priority 2 (ablation):** Add Trinity-no-CGate = 65 more runs. This tests whether the C-Gate specifically (not just dual agents) provides robustness.
- **Priority 3 (decoupling hypothesis):** Add Fused-LLM-RL = 65 more runs. This tests whether channel independence matters. Requires the fused model to be trained (P6-T2).

**LLM model choice:** The Analyst's pre-computed signals were generated with a specific LLM model. For final thesis results, switch to Claude 3.5 Sonnet for highest quality. Budget: ~$25–50 for all adversarial signal re-computation.

### Objective

Execute the full (or prioritized subset of) experiment matrix, verify completeness, and prepare raw results for analysis.

### Detailed Instructions

1. **Pre-flight checks:**
   - Verify all model files exist:
     - `experiments/executor/best_model/model.zip` (Executor)
     - `data/processed/precomputed_signals.json` (Analyst benign signals)
     - `data/processed/adversarial_signals_*.json` (adversarial Analyst signals — from P7-T2)
     - `experiments/executor/fused_model/model.zip` (if training is complete)
   - Verify all 12 attack YAML configs exist in `configs/adversarial/`
   - Run the mini experiment from P8-T1 checkpoint one more time to confirm stability

2. **Create the execution script** `scripts/run_full_evaluation.py`:

```python
"""
Execute the full experiment suite.

Usage:
    python scripts/run_full_evaluation.py --priority 1        # min viable (195 runs)
    python scripts/run_full_evaluation.py --priority 2        # + ablation (260 runs)
    python scripts/run_full_evaluation.py --priority 3        # full (325 runs)
    python scripts/run_full_evaluation.py --resume             # resume from last checkpoint
    python scripts/run_full_evaluation.py --pipeline Trinity --attack numeric_severe  # single combo
"""

import argparse
from pathlib import Path
from src.evaluation.runner import ExperimentRunner, ExperimentConfig

PRIORITY_1_PIPELINES = ["Trinity", "Executor-only", "Analyst-only"]
PRIORITY_2_PIPELINES = PRIORITY_1_PIPELINES + ["Trinity-no-CGate"]
PRIORITY_3_PIPELINES = PRIORITY_2_PIPELINES + ["Fused-LLM-RL"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have saved results")
    parser.add_argument("--pipeline", type=str, default=None,
                        help="Run only this pipeline")
    parser.add_argument("--attack", type=str, default=None,
                        help="Run only this attack condition")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.no_wandb:
        config.wandb_project = None

    runner = ExperimentRunner(config)

    # Determine which pipelines to run
    if args.pipeline:
        pipeline_names = [args.pipeline]
    elif args.priority == 1:
        pipeline_names = PRIORITY_1_PIPELINES
    elif args.priority == 2:
        pipeline_names = PRIORITY_2_PIPELINES
    else:
        pipeline_names = PRIORITY_3_PIPELINES

    # Filter available pipelines
    pipeline_names = [p for p in pipeline_names if p in runner.pipelines]

    # Determine attacks
    attack_names = [args.attack] if args.attack else list(runner.attack_configs.keys())

    # Determine seeds
    seeds = [args.seed] if args.seed else config.seeds

    # Run
    total = len(pipeline_names) * len(attack_names) * len(seeds)
    completed = 0
    failed = 0

    for pipeline_name in pipeline_names:
        for attack_name in attack_names:
            for seed in seeds:
                # Resume support: skip if result already exists
                if args.resume:
                    result_path = Path(config.output_dir) / pipeline_name / attack_name / f"seed_{seed}.json"
                    if result_path.exists():
                        completed += 1
                        continue

                print(f"[{completed+1}/{total}] {pipeline_name} / {attack_name} / seed={seed}")
                try:
                    runner.run_single(pipeline_name, attack_name, seed)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    failed += 1
                completed += 1

    print(f"\nDone: {completed} completed, {failed} failed out of {total} total")

if __name__ == "__main__":
    main()
```

3. **Execute in priority order:**

   **Priority 1 (~6.5 hours):**
   ```bash
   python scripts/run_full_evaluation.py --priority 1
   ```
   Monitor progress. If any runs fail, note the error and continue (the `--resume` flag lets you restart without re-running completed experiments).

   **After Priority 1 completes:**
   - Verify: `ls experiments/results/Trinity/` should show 13 subdirectories (benign + 12 attacks), each with 5 seed files.
   - Same for Executor-only and Analyst-only.
   - Quick sanity check: are Sharpe ratios in a reasonable range? Are there any NaN metrics?

   **Priority 2 (additional ~2 hours):**
   ```bash
   python scripts/run_full_evaluation.py --priority 2 --resume
   ```

   **Priority 3 (additional ~2 hours, only if fused model is trained):**
   ```bash
   python scripts/run_full_evaluation.py --priority 3 --resume
   ```

4. **Create a results aggregation script** `scripts/aggregate_results.py`:

```python
"""
Aggregate per-run JSON results into summary DataFrames.
Produces:
  - experiments/results/summary_by_condition.csv (mean +/- std across seeds)
  - experiments/results/summary_all_runs.csv (every individual run)
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

def aggregate():
    results_dir = Path("experiments/results")
    rows = []

    for pipeline_dir in sorted(results_dir.iterdir()):
        if not pipeline_dir.is_dir() or pipeline_dir.name.startswith("."):
            continue
        for attack_dir in sorted(pipeline_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            for seed_file in sorted(attack_dir.glob("seed_*.json")):
                with open(seed_file) as f:
                    result = json.load(f)
                if "error" in result:
                    continue
                row = {
                    "pipeline": result["pipeline"],
                    "attack": result["attack"],
                    "seed": result["seed"],
                    **result["metrics"],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary_all_runs.csv", index=False)

    # Aggregate by (pipeline, attack): mean +/- std
    summary = df.groupby(["pipeline", "attack"]).agg(
        pnl_mean=("pnl", "mean"), pnl_std=("pnl", "std"),
        sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
        mdd_mean=("max_drawdown", "mean"), mdd_std=("max_drawdown", "std"),
        var_mean=("var_95", "mean"), var_std=("var_95", "std"),
        es_mean=("es_95", "mean"), es_std=("es_95", "std"),
        n_runs=("seed", "count"),
    ).reset_index()
    summary.to_csv(results_dir / "summary_by_condition.csv", index=False)

    print(f"Aggregated {len(df)} runs into summary tables.")
    print(f"Pipelines: {df['pipeline'].unique().tolist()}")
    print(f"Attacks: {df['attack'].unique().tolist()}")
    print(f"Saved to {results_dir / 'summary_by_condition.csv'}")

if __name__ == "__main__":
    aggregate()
```

5. **Run aggregation and verify completeness:**
   ```bash
   python scripts/aggregate_results.py
   ```
   Check:
   - `summary_all_runs.csv` should have 325 rows (or however many were run)
   - `summary_by_condition.csv` should have 5x13 = 65 rows
   - No missing combinations (unless deliberately skipped)

### Acceptance Criteria

- [ ] `scripts/run_full_evaluation.py` exists with priority levels and resume support
- [ ] `scripts/aggregate_results.py` exists and produces summary CSVs
- [ ] Priority 1 runs completed: 3 systems x 13 conditions x 5 seeds = 195 runs (minimum)
- [ ] No more than 5% of runs failed (<=10 failures out of 195)
- [ ] `experiments/results/summary_all_runs.csv` exists with all individual run metrics
- [ ] `experiments/results/summary_by_condition.csv` exists with aggregated (mean +/- std) metrics
- [ ] W&B dashboard is populated with all completed runs
- [ ] Quick sanity check: metrics are in reasonable ranges, no all-NaN columns

### Files to Create/Modify

- **Create:** `scripts/run_full_evaluation.py`
- **Create:** `scripts/aggregate_results.py`
- **Output:** `experiments/results/summary_all_runs.csv`
- **Output:** `experiments/results/summary_by_condition.csv`
- **Output:** `experiments/results/{pipeline}/{attack}/seed_{n}.json` (325 files)

### Dependencies

- P8-T1 (ExperimentRunner)
- All Phase 6 and 7 components

### Human Checkpoint

Before proceeding to P8-T3:
1. **Completeness check:** Run `aggregate_results.py` and verify row counts. Any missing runs?
2. **Spot-check results:** Pick 3 random runs and inspect their JSON output. Do the daily PnL values look reasonable? Are positions changing or stuck at one value?
3. **Preliminary signal:** Look at Trinity vs. Executor-only Sharpe ratio under benign vs. severe attacks. Is there a visible trend? If Trinity doesn't degrade less than Executor-only, there may be an issue with the C-Gate or Guardian configuration.
4. **Budget check:** If you need to re-run with Claude Sonnet signals, confirm the cost estimate and approve before proceeding.

---

## P8-T3: C-Gate Behavior Analysis and Threshold Calibration

**Estimated time:** ~3 hours
**Dependencies:** P8-T2 complete (full experiment results available in `experiments/results/`)

### Context

The C-Gate is the central mechanism of the Trinity architecture. It computes a divergence score Delta between the Executor's distribution pi_RL and the Analyst's distribution pi_LLM, then classifies each timestep into one of three regimes based on thresholds tau_low and tau_high:
- **Agreement** (Delta < tau_low): Both agents agree; blend their distributions.
- **Ambiguity** (tau_low <= Delta < tau_high): Moderate disagreement; use a cautious weighted blend.
- **Conflict** (Delta >= tau_high): Strong disagreement; defer to the uncompromised channel.

The thesis hypothesis predicts that under adversarial attack on one channel, Delta should increase (the attacked channel diverges from the unattacked one), pushing the C-Gate into conflict regime and effectively isolating the compromised signal. This task analyzes whether this actually happens in the experimental data.

Key questions:
1. **Does Delta increase under attack?** Compare Delta distributions benign vs. each attack type.
2. **Do regime frequencies shift as expected?** More conflict under attack, more agreement under benign.
3. **Are the current thresholds optimal?** Sweep tau_low and tau_high to find best robustness-performance tradeoff.
4. **Can Delta serve as an attack detector?** ROC analysis of Delta as a binary classifier for "under attack."

### Objective

Analyze C-Gate behavior across all experimental conditions, calibrate optimal thresholds, and produce publication-ready visualizations of the C-Gate's role in robustness.

### Detailed Instructions

1. **Create `scripts/analyze_cgate.py`:**

```python
"""
Analyze C-Gate behavior across experimental conditions.

Produces:
  - Delta distribution histograms (benign vs. each attack)
  - Regime frequency tables
  - Threshold sensitivity heatmap
  - ROC curves for attack detection via Delta

Usage:
    python scripts/analyze_cgate.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

RESULTS_DIR = Path("experiments/results")
OUTPUT_DIR = Path("experiments/analysis/cgate")


def load_trinity_results() -> dict[str, list[dict]]:
    """
    Load all Trinity results (only Trinity has C-Gate data).
    Returns dict: attack_name -> list of run dicts (one per seed).
    """
    trinity_dir = RESULTS_DIR / "Trinity"
    results = defaultdict(list)
    for attack_dir in sorted(trinity_dir.iterdir()):
        if not attack_dir.is_dir():
            continue
        for seed_file in sorted(attack_dir.glob("seed_*.json")):
            with open(seed_file) as f:
                data = json.load(f)
            if "error" not in data:
                results[attack_dir.name].append(data)
    return dict(results)


def extract_deltas(results: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """
    Extract all Delta values per condition.
    Returns dict: attack_name -> array of all Delta values across seeds.
    """
    deltas = {}
    for attack_name, runs in results.items():
        all_deltas = []
        for run in runs:
            if "deltas" in run.get("time_series", {}):
                run_deltas = run["time_series"]["deltas"]
                all_deltas.extend([d for d in run_deltas if d is not None])
        deltas[attack_name] = np.array(all_deltas) if all_deltas else np.array([])
    return deltas


def plot_delta_distributions(deltas: dict[str, np.ndarray]):
    """
    Plot overlapping histograms of Delta distributions:
    benign vs. each attack type.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benign_deltas = deltas.get("benign", np.array([]))

    attack_categories = {
        "numeric": [k for k in deltas if k.startswith("numeric_")],
        "semantic": [k for k in deltas if k.startswith("semantic_")],
        "coordinated": [k for k in deltas if k.startswith("coordinated_")],
    }

    for category, attack_names in attack_categories.items():
        if not attack_names:
            continue

        fig, axes = plt.subplots(1, len(attack_names), figsize=(6 * len(attack_names), 4),
                                  sharey=True)
        if len(attack_names) == 1:
            axes = [axes]

        for ax, attack_name in zip(axes, sorted(attack_names)):
            attack_deltas = deltas.get(attack_name, np.array([]))
            if len(benign_deltas) > 0:
                ax.hist(benign_deltas, bins=50, alpha=0.5, density=True,
                        label="Benign", color="steelblue")
            if len(attack_deltas) > 0:
                ax.hist(attack_deltas, bins=50, alpha=0.5, density=True,
                        label=attack_name, color="indianred")
            ax.set_xlabel("Delta (divergence)")
            ax.set_ylabel("Density")
            ax.set_title(f"Benign vs. {attack_name}")
            ax.legend()

        fig.suptitle(f"Delta Distributions: {category.title()} Attacks", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"delta_distribution_{category}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)


def compute_regime_frequencies(results: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Compute regime frequency table: rows = attack conditions, cols = regimes.
    """
    rows = []
    for attack_name, runs in results.items():
        all_regimes = []
        for run in runs:
            if "regimes" in run.get("time_series", {}):
                all_regimes.extend(run["time_series"]["regimes"])
        if not all_regimes:
            continue

        total = len(all_regimes)
        counts = {
            "agreement": sum(1 for r in all_regimes if r == "agreement"),
            "ambiguity": sum(1 for r in all_regimes if r == "ambiguity"),
            "conflict": sum(1 for r in all_regimes if r == "conflict"),
        }
        rows.append({
            "attack": attack_name,
            "agreement_pct": counts["agreement"] / total * 100,
            "ambiguity_pct": counts["ambiguity"] / total * 100,
            "conflict_pct": counts["conflict"] / total * 100,
            "n_timesteps": total,
        })

    df = pd.DataFrame(rows).sort_values("attack").reset_index(drop=True)
    return df


def threshold_sensitivity_heatmap(
    deltas: dict[str, np.ndarray],
    results: dict[str, list[dict]],
    tau_low_range: np.ndarray = np.linspace(0.05, 0.5, 10),
    tau_high_range: np.ndarray = np.linspace(0.3, 1.0, 10),
):
    """
    Sweep tau_low and tau_high, recompute regime assignments and a robustness
    score for each (tau_low, tau_high) pair. Produce a heatmap.

    Robustness score = mean Sharpe degradation reduction for Trinity vs. Executor-only.
    Simplified version: for each threshold pair, compute fraction of attack timesteps
    correctly routed to conflict regime.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all attack deltas (non-benign)
    attack_deltas = np.concatenate([
        d for name, d in deltas.items() if name != "benign" and len(d) > 0
    ])
    benign_deltas_arr = deltas.get("benign", np.array([]))

    if len(attack_deltas) == 0 or len(benign_deltas_arr) == 0:
        print("WARNING: Insufficient delta data for threshold sensitivity analysis.")
        return

    # For each (tau_low, tau_high), compute:
    # - attack_conflict_rate: fraction of attack timesteps in conflict (want high)
    # - benign_agreement_rate: fraction of benign timesteps in agreement (want high)
    # - combined_score = attack_conflict_rate + benign_agreement_rate (max = 2.0)
    score_matrix = np.zeros((len(tau_low_range), len(tau_high_range)))

    for i, tau_low in enumerate(tau_low_range):
        for j, tau_high in enumerate(tau_high_range):
            if tau_low >= tau_high:
                score_matrix[i, j] = np.nan
                continue

            attack_conflict = np.mean(attack_deltas >= tau_high)
            benign_agreement = np.mean(benign_deltas_arr < tau_low)
            score_matrix[i, j] = attack_conflict + benign_agreement

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.isnan(score_matrix)
    sns.heatmap(
        score_matrix,
        xticklabels=[f"{t:.2f}" for t in tau_high_range],
        yticklabels=[f"{t:.2f}" for t in tau_low_range],
        annot=True, fmt=".2f", cmap="YlOrRd", mask=mask, ax=ax,
    )
    ax.set_xlabel("tau_high (conflict threshold)")
    ax.set_ylabel("tau_low (agreement threshold)")
    ax.set_title("Threshold Sensitivity: Combined Score\n"
                  "(attack conflict rate + benign agreement rate, max=2.0)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_sensitivity_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Find optimal thresholds
    valid_mask = ~np.isnan(score_matrix)
    if valid_mask.any():
        best_idx = np.unravel_index(np.nanargmax(score_matrix), score_matrix.shape)
        best_tau_low = tau_low_range[best_idx[0]]
        best_tau_high = tau_high_range[best_idx[1]]
        best_score = score_matrix[best_idx]
        print(f"Optimal thresholds: tau_low={best_tau_low:.3f}, tau_high={best_tau_high:.3f}")
        print(f"Combined score: {best_score:.3f} / 2.0")

        # Save to JSON
        with open(OUTPUT_DIR / "optimal_thresholds.json", "w") as f:
            json.dump({
                "tau_low": float(best_tau_low),
                "tau_high": float(best_tau_high),
                "combined_score": float(best_score),
                "attack_conflict_rate": float(np.mean(attack_deltas >= best_tau_high)),
                "benign_agreement_rate": float(np.mean(benign_deltas_arr < best_tau_low)),
            }, f, indent=2)


def roc_analysis(deltas: dict[str, np.ndarray]):
    """
    Treat Delta as a binary classifier: can it distinguish 'under attack' from 'benign'?
    For each timestep, label = 1 if from an attack run, 0 if from benign.
    Score = Delta value. Compute ROC curve and AUC.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benign_deltas_arr = deltas.get("benign", np.array([]))
    if len(benign_deltas_arr) == 0:
        print("WARNING: No benign deltas for ROC analysis.")
        return

    # Group attacks by category for separate ROC curves
    attack_categories = {
        "numeric": [k for k in deltas if k.startswith("numeric_")],
        "semantic": [k for k in deltas if k.startswith("semantic_")],
        "coordinated": [k for k in deltas if k.startswith("coordinated_")],
        "all_attacks": [k for k in deltas if k != "benign"],
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"numeric": "tab:blue", "semantic": "tab:orange",
              "coordinated": "tab:red", "all_attacks": "black"}

    roc_results = {}

    for category, attack_names in attack_categories.items():
        attack_deltas = np.concatenate([
            deltas[name] for name in attack_names if len(deltas.get(name, [])) > 0
        ])
        if len(attack_deltas) == 0:
            continue

        # Labels: 0 for benign, 1 for attack
        y_true = np.concatenate([
            np.zeros(len(benign_deltas_arr)),
            np.ones(len(attack_deltas)),
        ])
        y_score = np.concatenate([benign_deltas_arr, attack_deltas])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors.get(category, "gray"),
                label=f"{category} (AUC={roc_auc:.3f})")

        # Find optimal threshold (Youden's J statistic)
        j_statistic = tpr - fpr
        optimal_idx = np.argmax(j_statistic)
        optimal_threshold = thresholds[optimal_idx]

        roc_results[category] = {
            "auc": float(roc_auc),
            "optimal_threshold": float(optimal_threshold),
            "tpr_at_optimal": float(tpr[optimal_idx]),
            "fpr_at_optimal": float(fpr[optimal_idx]),
        }

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Delta as Attack Detector")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_attack_detection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save ROC results
    with open(OUTPUT_DIR / "roc_results.json", "w") as f:
        json.dump(roc_results, f, indent=2)

    for cat, res in roc_results.items():
        print(f"  {cat}: AUC={res['auc']:.3f}, optimal threshold={res['optimal_threshold']:.4f}")


def main():
    print("Loading Trinity results...")
    results = load_trinity_results()
    print(f"  Loaded {sum(len(v) for v in results.values())} runs across {len(results)} conditions")

    print("\nExtracting Delta values...")
    deltas = extract_deltas(results)
    for name, d in deltas.items():
        print(f"  {name}: {len(d)} Delta values, mean={d.mean():.4f}, std={d.std():.4f}" if len(d) > 0 else f"  {name}: no data")

    print("\nPlotting Delta distributions...")
    plot_delta_distributions(deltas)

    print("\nComputing regime frequencies...")
    regime_df = compute_regime_frequencies(results)
    print(regime_df.to_string(index=False))
    regime_df.to_csv(OUTPUT_DIR / "regime_frequencies.csv", index=False)

    print("\nRunning threshold sensitivity analysis...")
    threshold_sensitivity_heatmap(deltas, results)

    print("\nRunning ROC analysis...")
    roc_analysis(deltas)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

2. **Create the analysis notebook** `notebooks/cgate_analysis.ipynb` (optional, for interactive exploration):

   The notebook should contain the same analyses as the script above but in interactive form, allowing the researcher to:
   - Zoom into specific attack types
   - Overlay multiple Delta distributions
   - Interactively adjust thresholds and see regime reassignments
   - Correlate Delta spikes with price events

3. **Key analyses to perform and document:**

   **a) Delta distribution shift:**
   - For each attack type, compute the Kolmogorov-Smirnov statistic between benign and attack Delta distributions.
   - Report mean Delta shift: `mean(Delta_attack) - mean(Delta_benign)`.
   - Expected: numeric attacks cause largest Delta shift (they directly distort the Executor's input), semantic attacks cause moderate shift (Analyst output changes), coordinated attacks may show complex bimodal patterns.

   **b) Regime frequency table:**
   - Expected benign: high agreement (>60%), low conflict (<10%).
   - Expected under severe numeric attack: agreement drops, conflict rises significantly.
   - Expected under semantic attack: similar pattern but less pronounced if Analyst signal is weaker.
   - If conflict does NOT increase under attack, investigate: is the C-Gate threshold too high? Is the Analyst not providing enough signal?

   **c) Threshold calibration results:**
   - Report the optimal (tau_low, tau_high) pair and its combined score.
   - Compare with the default thresholds from the config. If the optimal thresholds differ significantly, note this as a finding.
   - **Important:** The optimal thresholds should not be tuned on the test set. If you plan to use calibrated thresholds in the final evaluation, use a validation split or cross-validation.

   **d) ROC analysis for attack detection:**
   - AUC > 0.8 means Delta is a reliable attack indicator.
   - AUC near 0.5 means Delta cannot distinguish attack from benign (C-Gate is not detecting the attack).
   - Report per-category AUC and the overall AUC.

### Acceptance Criteria

- [ ] `scripts/analyze_cgate.py` runs end-to-end and produces all outputs
- [ ] Delta distribution histograms saved to `experiments/analysis/cgate/delta_distribution_*.png`
- [ ] Regime frequency table saved to `experiments/analysis/cgate/regime_frequencies.csv`
- [ ] Threshold sensitivity heatmap saved to `experiments/analysis/cgate/threshold_sensitivity_heatmap.png`
- [ ] Optimal thresholds saved to `experiments/analysis/cgate/optimal_thresholds.json`
- [ ] ROC curves and AUC values saved to `experiments/analysis/cgate/roc_attack_detection.png` and `roc_results.json`
- [ ] All figures are 300 DPI, publication-ready with proper labels and legends
- [ ] Key findings documented: Does Delta increase under attack? Which attack type causes the largest shift? Are thresholds well-calibrated?

### Files to Create/Modify

- **Create:** `scripts/analyze_cgate.py`
- **Create (optional):** `notebooks/cgate_analysis.ipynb`
- **Output:** `experiments/analysis/cgate/delta_distribution_numeric.png`
- **Output:** `experiments/analysis/cgate/delta_distribution_semantic.png`
- **Output:** `experiments/analysis/cgate/delta_distribution_coordinated.png`
- **Output:** `experiments/analysis/cgate/regime_frequencies.csv`
- **Output:** `experiments/analysis/cgate/threshold_sensitivity_heatmap.png`
- **Output:** `experiments/analysis/cgate/optimal_thresholds.json`
- **Output:** `experiments/analysis/cgate/roc_attack_detection.png`
- **Output:** `experiments/analysis/cgate/roc_results.json`

### Dependencies

- P8-T2 (experiment results, specifically Trinity runs with Delta and regime data)
- `scikit-learn` for ROC analysis
- `seaborn` for heatmaps

### Human Checkpoint

Before proceeding to P8-T4:
1. **Examine the Delta distributions.** Is there a clear separation between benign and attack? If not, the C-Gate may not be functioning as the thesis claims.
2. **Check regime frequencies.** Does conflict increase under attack? If the regime distribution is nearly identical across conditions, the thresholds may need recalibrating, or the Delta metric is not sensitive enough.
3. **Review the optimal thresholds.** If they differ substantially from the defaults, decide whether to re-run the full experiment with calibrated thresholds (this requires a validation set, not the test set, to avoid data leakage).
4. **ROC AUC sanity check.** If AUC < 0.6 for all categories, there is a fundamental issue with the C-Gate's ability to detect adversarial input. This would require revisiting the architecture, not just the thresholds.

---

## P8-T4: Generate Thesis-Ready Results Tables and Figures

**Estimated time:** ~5 hours
**Dependencies:** P8-T2 (aggregated results CSVs), P8-T3 (C-Gate analysis outputs)

### Context

The thesis requires publication-quality tables and figures that present the experimental results clearly. These must follow academic standards: proper axis labels, legends, statistical annotations (mean +/- std), and consistent styling. All figures should be saved as both PNG (300 DPI for quick review) and PDF (vector format for LaTeX inclusion).

The key results to present:
1. **Benign performance table:** How do the 5 systems perform without any attack? This is the baseline.
2. **Attack degradation matrix:** How much does each system degrade under each attack type? This is the core thesis result.
3. **C-Gate regime frequency table:** How does the C-Gate behave across conditions? This demonstrates the mechanism.
4. **Cumulative PnL curves:** Visual comparison of system performance over time.
5. **Delta violin plots:** Distribution of Delta values across conditions.
6. **Degradation heatmaps:** Visual summary of the degradation matrix.

### Objective

Generate all tables and figures needed for the thesis results chapter. Ensure they are reproducible from the raw experimental data via scripts.

### Detailed Instructions

1. **Create `scripts/generate_tables.py`:**

```python
"""
Generate LaTeX-formatted tables for the thesis results chapter.

Produces:
  - experiments/tables/benign_performance.tex
  - experiments/tables/attack_degradation.tex
  - experiments/tables/cgate_regimes.tex
  - experiments/tables/flip_rates.tex

Usage:
    python scripts/generate_tables.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
CGATE_DIR = Path("experiments/analysis/cgate")
OUTPUT_DIR = Path("experiments/tables")


def format_metric(mean: float, std: float, fmt: str = ".2f") -> str:
    """Format as 'mean +/- std' for LaTeX."""
    return f"${mean:{fmt}} \\pm {std:{fmt}}$"


def generate_benign_performance_table():
    """
    Table 1: Benign performance across all 5 systems.
    Columns: System | PnL ($) | Sharpe | Max DD (%) | VaR 95% ($) | ES 95% ($)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")
    benign = df[df["attack"] == "benign"]

    rows = []
    pipeline_order = ["Trinity", "Executor-only", "Analyst-only",
                      "Trinity-no-CGate", "Fused-LLM-RL"]

    for pipeline in pipeline_order:
        p_data = benign[benign["pipeline"] == pipeline]
        if p_data.empty:
            continue
        rows.append({
            "System": pipeline,
            "PnL (\\$)": format_metric(p_data["pnl"].mean(), p_data["pnl"].std(), ".0f"),
            "Sharpe": format_metric(p_data["sharpe"].mean(), p_data["sharpe"].std(), ".3f"),
            "Max DD (\\%)": format_metric(
                p_data["max_drawdown"].mean() * 100, p_data["max_drawdown"].std() * 100, ".1f"),
            "VaR$_{95}$ (\\$)": format_metric(
                p_data["var_95"].mean(), p_data["var_95"].std(), ".0f"),
            "ES$_{95}$ (\\$)": format_metric(
                p_data["es_95"].mean(), p_data["es_95"].std(), ".0f"),
        })

    table_df = pd.DataFrame(rows)

    # Generate LaTeX
    latex = table_df.to_latex(
        index=False, escape=False, column_format="l" + "r" * (len(table_df.columns) - 1),
        caption="Benign performance of all trading systems (mean $\\pm$ std over 5 seeds).",
        label="tab:benign_performance",
    )
    with open(OUTPUT_DIR / "benign_performance.tex", "w") as f:
        f.write(latex)
    print(f"  Saved {OUTPUT_DIR / 'benign_performance.tex'}")


def generate_attack_degradation_table():
    """
    Table 2: Sharpe ratio degradation (%) under each attack.
    Rows = attack conditions, Columns = systems.
    Values = percentage Sharpe degradation relative to benign.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")
    benign = df[df["attack"] == "benign"]

    pipeline_order = ["Trinity", "Executor-only", "Analyst-only",
                      "Trinity-no-CGate", "Fused-LLM-RL"]
    available_pipelines = [p for p in pipeline_order if p in df["pipeline"].unique()]

    # Compute benign baseline Sharpe per pipeline per seed
    benign_sharpe = {}
    for pipeline in available_pipelines:
        p_benign = benign[benign["pipeline"] == pipeline]
        # Map seed -> benign sharpe
        benign_sharpe[pipeline] = dict(zip(p_benign["seed"], p_benign["sharpe"]))

    attack_names = sorted([a for a in df["attack"].unique() if a != "benign"])
    rows = []

    for attack in attack_names:
        row = {"Attack": attack.replace("_", " ").title()}
        attack_data = df[df["attack"] == attack]

        for pipeline in available_pipelines:
            p_attack = attack_data[attack_data["pipeline"] == pipeline]
            if p_attack.empty:
                row[pipeline] = "---"
                continue

            # Compute per-seed degradation, then mean +/- std
            degradations = []
            for _, run in p_attack.iterrows():
                b_sharpe = benign_sharpe.get(pipeline, {}).get(run["seed"])
                if b_sharpe is not None and abs(b_sharpe) > 1e-10:
                    deg = (b_sharpe - run["sharpe"]) / abs(b_sharpe) * 100
                    degradations.append(deg)

            if degradations:
                mean_deg = np.mean(degradations)
                std_deg = np.std(degradations)
                row[pipeline] = format_metric(mean_deg, std_deg, ".1f")
            else:
                row[pipeline] = "---"

        rows.append(row)

    table_df = pd.DataFrame(rows)
    latex = table_df.to_latex(
        index=False, escape=False,
        column_format="l" + "r" * len(available_pipelines),
        caption="Sharpe ratio degradation (\\%) under adversarial attacks "
                "(positive = worse, mean $\\pm$ std over 5 seeds).",
        label="tab:attack_degradation",
    )
    with open(OUTPUT_DIR / "attack_degradation.tex", "w") as f:
        f.write(latex)
    print(f"  Saved {OUTPUT_DIR / 'attack_degradation.tex'}")


def generate_cgate_regime_table():
    """
    Table 3: C-Gate regime frequencies (%) across conditions.
    Uses regime_frequencies.csv from P8-T3.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_csv = CGATE_DIR / "regime_frequencies.csv"
    if not regime_csv.exists():
        print("  WARNING: regime_frequencies.csv not found. Run P8-T3 first.")
        return

    df = pd.read_csv(regime_csv)

    # Rename columns for LaTeX
    df = df.rename(columns={
        "attack": "Condition",
        "agreement_pct": "Agreement (\\%)",
        "ambiguity_pct": "Ambiguity (\\%)",
        "conflict_pct": "Conflict (\\%)",
    })
    df["Condition"] = df["Condition"].str.replace("_", " ").str.title()
    df = df[["Condition", "Agreement (\\%)", "Ambiguity (\\%)", "Conflict (\\%)"]]

    # Format percentages
    for col in ["Agreement (\\%)", "Ambiguity (\\%)", "Conflict (\\%)"]:
        df[col] = df[col].apply(lambda x: f"{x:.1f}")

    latex = df.to_latex(
        index=False, escape=False, column_format="lrrr",
        caption="C-Gate regime frequencies (\\%) across experimental conditions.",
        label="tab:cgate_regimes",
    )
    with open(OUTPUT_DIR / "cgate_regimes.tex", "w") as f:
        f.write(latex)
    print(f"  Saved {OUTPUT_DIR / 'cgate_regimes.tex'}")


def generate_flip_rate_table():
    """
    Table 4: Action flip rates under each attack.
    Rows = attack conditions, Columns = systems.
    Values = flip rate (fraction of changed actions).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")

    pipeline_order = ["Trinity", "Executor-only", "Analyst-only",
                      "Trinity-no-CGate", "Fused-LLM-RL"]
    available_pipelines = [p for p in pipeline_order if p in df["pipeline"].unique()]

    # Load per-run time series to compute flip rates
    from src.evaluation.robustness import compute_flip_rate

    attack_names = sorted([a for a in df["attack"].unique() if a != "benign"])
    rows = []

    for attack in attack_names:
        row = {"Attack": attack.replace("_", " ").title()}

        for pipeline in available_pipelines:
            flip_rates = []
            for seed in df["seed"].unique():
                benign_file = RESULTS_DIR / pipeline / "benign" / f"seed_{seed}.json"
                attack_file = RESULTS_DIR / pipeline / attack / f"seed_{seed}.json"

                if not benign_file.exists() or not attack_file.exists():
                    continue

                with open(benign_file) as f:
                    benign_data = json.load(f)
                with open(attack_file) as f:
                    attack_data = json.load(f)

                if "error" in benign_data or "error" in attack_data:
                    continue

                actions_b = benign_data["time_series"]["actions"]
                actions_a = attack_data["time_series"]["actions"]

                if len(actions_b) == len(actions_a):
                    fr = compute_flip_rate(actions_b, actions_a)
                    flip_rates.append(fr)

            if flip_rates:
                mean_fr = np.mean(flip_rates)
                std_fr = np.std(flip_rates)
                row[pipeline] = format_metric(mean_fr * 100, std_fr * 100, ".1f")
            else:
                row[pipeline] = "---"

        rows.append(row)

    table_df = pd.DataFrame(rows)
    latex = table_df.to_latex(
        index=False, escape=False,
        column_format="l" + "r" * len(available_pipelines),
        caption="Action flip rates (\\%) under adversarial attacks "
                "(fraction of actions that changed vs. benign, mean $\\pm$ std over 5 seeds).",
        label="tab:flip_rates",
    )
    with open(OUTPUT_DIR / "flip_rates.tex", "w") as f:
        f.write(latex)
    print(f"  Saved {OUTPUT_DIR / 'flip_rates.tex'}")


def main():
    print("Generating thesis tables...")
    print("\n1. Benign performance table:")
    generate_benign_performance_table()
    print("\n2. Attack degradation table:")
    generate_attack_degradation_table()
    print("\n3. C-Gate regime table:")
    generate_cgate_regime_table()
    print("\n4. Flip rate table:")
    generate_flip_rate_table()
    print(f"\nAll tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

2. **Create `scripts/generate_figures.py`:**

```python
"""
Generate thesis-ready figures from experimental results.

Produces:
  - experiments/figures/cumulative_pnl_benign.{png,pdf}
  - experiments/figures/cumulative_pnl_under_attack.{png,pdf}
  - experiments/figures/delta_violins.{png,pdf}
  - experiments/figures/degradation_heatmap.{png,pdf}
  - experiments/figures/flip_rate_bars.{png,pdf}

Usage:
    python scripts/generate_figures.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
OUTPUT_DIR = Path("experiments/figures")

# Consistent styling
PIPELINE_COLORS = {
    "Trinity": "#2ecc71",
    "Executor-only": "#3498db",
    "Analyst-only": "#e67e22",
    "Trinity-no-CGate": "#9b59b6",
    "Fused-LLM-RL": "#e74c3c",
}
PIPELINE_ORDER = ["Trinity", "Executor-only", "Analyst-only",
                  "Trinity-no-CGate", "Fused-LLM-RL"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_figure(fig, name: str):
    """Save figure in both PNG and PDF formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png and {name}.pdf")


def plot_cumulative_pnl_benign():
    """
    Figure 1: Cumulative PnL curves for all systems under benign conditions.
    Shows median seed with shaded min-max band.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for pipeline in PIPELINE_ORDER:
        pnl_dir = RESULTS_DIR / pipeline / "benign"
        if not pnl_dir.exists():
            continue

        all_cum_pnl = []
        dates = None
        for seed_file in sorted(pnl_dir.glob("seed_*.json")):
            with open(seed_file) as f:
                data = json.load(f)
            if "error" in data:
                continue
            cum_pnl = data["time_series"]["cumulative_pnl"]
            all_cum_pnl.append(cum_pnl)
            if dates is None:
                dates = data["dates"]

        if not all_cum_pnl or dates is None:
            continue

        # Align to same length
        min_len = min(len(p) for p in all_cum_pnl)
        all_cum_pnl = [p[:min_len] for p in all_cum_pnl]
        dates = dates[:min_len]

        arr = np.array(all_cum_pnl)
        median = np.median(arr, axis=0)
        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)

        color = PIPELINE_COLORS.get(pipeline, "gray")
        x = range(len(median))
        ax.plot(x, median, label=pipeline, color=color, linewidth=2)
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Cumulative PnL Under Benign Conditions")
    ax.legend(loc="upper left")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    save_figure(fig, "cumulative_pnl_benign")


def plot_cumulative_pnl_under_attack():
    """
    Figure 2: Cumulative PnL for Trinity vs. Executor-only under selected attacks.
    One subplot per attack severity (low, moderate, severe) for numeric attacks.
    """
    attack_levels = ["numeric_low", "numeric_moderate", "numeric_severe"]
    compare_pipelines = ["Trinity", "Executor-only"]

    fig, axes = plt.subplots(1, len(attack_levels), figsize=(6 * len(attack_levels), 5),
                              sharey=True)
    if len(attack_levels) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attack_levels):
        for pipeline in compare_pipelines:
            pnl_dir = RESULTS_DIR / pipeline / attack
            if not pnl_dir.exists():
                continue

            all_cum_pnl = []
            for seed_file in sorted(pnl_dir.glob("seed_*.json")):
                with open(seed_file) as f:
                    data = json.load(f)
                if "error" in data:
                    continue
                all_cum_pnl.append(data["time_series"]["cumulative_pnl"])

            if not all_cum_pnl:
                continue

            min_len = min(len(p) for p in all_cum_pnl)
            arr = np.array([p[:min_len] for p in all_cum_pnl])
            median = np.median(arr, axis=0)
            lo = np.min(arr, axis=0)
            hi = np.max(arr, axis=0)

            color = PIPELINE_COLORS.get(pipeline, "gray")
            x = range(len(median))
            ax.plot(x, median, label=pipeline, color=color, linewidth=2)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)

        ax.set_xlabel("Trading Day")
        ax.set_title(attack.replace("_", " ").title())
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[-1].legend(loc="upper left")
    fig.suptitle("Trinity vs. Executor-Only Under Numeric Attacks", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "cumulative_pnl_under_attack")


def plot_delta_violins():
    """
    Figure 3: Violin plots of Delta distributions across conditions.
    X-axis = attack condition, Y-axis = Delta value.
    """
    trinity_dir = RESULTS_DIR / "Trinity"
    if not trinity_dir.exists():
        print("  WARNING: No Trinity results for violin plots.")
        return

    records = []
    for attack_dir in sorted(trinity_dir.iterdir()):
        if not attack_dir.is_dir():
            continue
        for seed_file in attack_dir.glob("seed_*.json"):
            with open(seed_file) as f:
                data = json.load(f)
            if "error" in data or "deltas" not in data.get("time_series", {}):
                continue
            for d in data["time_series"]["deltas"]:
                if d is not None:
                    records.append({"condition": attack_dir.name, "delta": d})

    if not records:
        print("  WARNING: No Delta data for violin plots.")
        return

    df = pd.DataFrame(records)

    # Order conditions: benign first, then sorted attacks
    conditions = ["benign"] + sorted([c for c in df["condition"].unique() if c != "benign"])
    df["condition"] = pd.Categorical(df["condition"], categories=conditions, ordered=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.violinplot(data=df, x="condition", y="delta", ax=ax, cut=0,
                   palette="coolwarm", inner="quartile")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Delta (Divergence Score)")
    ax.set_title("C-Gate Delta Distributions Across Conditions")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, "delta_violins")


def plot_degradation_heatmap():
    """
    Figure 4: Heatmap of Sharpe ratio degradation (%).
    Rows = attacks, Columns = systems. Color = degradation magnitude.
    """
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")
    benign = df[df["attack"] == "benign"]

    available_pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
    attack_names = sorted([a for a in df["attack"].unique() if a != "benign"])

    # Build degradation matrix
    degradation_matrix = np.full((len(attack_names), len(available_pipelines)), np.nan)

    for i, attack in enumerate(attack_names):
        for j, pipeline in enumerate(available_pipelines):
            b_sharpe = benign[benign["pipeline"] == pipeline]["sharpe"].mean()
            a_sharpe = df[(df["attack"] == attack) & (df["pipeline"] == pipeline)]["sharpe"].mean()
            if abs(b_sharpe) > 1e-10:
                degradation_matrix[i, j] = (b_sharpe - a_sharpe) / abs(b_sharpe) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        degradation_matrix,
        xticklabels=available_pipelines,
        yticklabels=[a.replace("_", " ").title() for a in attack_names],
        annot=True, fmt=".1f", cmap="RdYlGn_r", center=0, ax=ax,
        cbar_kws={"label": "Sharpe Degradation (%)"},
    )
    ax.set_title("Sharpe Ratio Degradation Under Adversarial Attacks (%)")
    ax.set_xlabel("Trading System")
    ax.set_ylabel("Attack Condition")
    save_figure(fig, "degradation_heatmap")


def plot_flip_rate_bars():
    """
    Figure 5: Grouped bar chart of flip rates per system per attack category.
    """
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")
    available_pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]

    from src.evaluation.robustness import compute_flip_rate

    # Compute flip rates grouped by attack category
    attack_categories = {}
    for attack in df["attack"].unique():
        if attack == "benign":
            continue
        category = attack.split("_")[0]  # numeric, semantic, coordinated
        if category not in attack_categories:
            attack_categories[category] = []
        attack_categories[category].append(attack)

    records = []
    for category, attacks in sorted(attack_categories.items()):
        for pipeline in available_pipelines:
            all_flip_rates = []
            for attack in attacks:
                for seed in df["seed"].unique():
                    benign_file = RESULTS_DIR / pipeline / "benign" / f"seed_{seed}.json"
                    attack_file = RESULTS_DIR / pipeline / attack / f"seed_{seed}.json"
                    if not benign_file.exists() or not attack_file.exists():
                        continue
                    with open(benign_file) as f:
                        bd = json.load(f)
                    with open(attack_file) as f:
                        ad = json.load(f)
                    if "error" in bd or "error" in ad:
                        continue
                    ab = bd["time_series"]["actions"]
                    aa = ad["time_series"]["actions"]
                    if len(ab) == len(aa):
                        all_flip_rates.append(compute_flip_rate(ab, aa))

            if all_flip_rates:
                records.append({
                    "category": category.title(),
                    "pipeline": pipeline,
                    "flip_rate_mean": np.mean(all_flip_rates) * 100,
                    "flip_rate_std": np.std(all_flip_rates) * 100,
                })

    if not records:
        print("  WARNING: No flip rate data for bar chart.")
        return

    plot_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = sorted(plot_df["category"].unique())
    x = np.arange(len(categories))
    width = 0.15
    n_pipelines = len(available_pipelines)

    for i, pipeline in enumerate(available_pipelines):
        p_data = plot_df[plot_df["pipeline"] == pipeline]
        means = [p_data[p_data["category"] == c]["flip_rate_mean"].values[0]
                 if len(p_data[p_data["category"] == c]) > 0 else 0
                 for c in categories]
        stds = [p_data[p_data["category"] == c]["flip_rate_std"].values[0]
                if len(p_data[p_data["category"] == c]) > 0 else 0
                for c in categories]
        offset = (i - n_pipelines / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=pipeline, color=PIPELINE_COLORS.get(pipeline, "gray"),
               capsize=3, alpha=0.85)

    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Action Flip Rate (%)")
    ax.set_title("Action Stability Under Adversarial Attacks")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, "flip_rate_bars")


def main():
    print("Generating thesis figures...")
    print("\n1. Cumulative PnL (benign):")
    plot_cumulative_pnl_benign()
    print("\n2. Cumulative PnL (under attack):")
    plot_cumulative_pnl_under_attack()
    print("\n3. Delta violin plots:")
    plot_delta_violins()
    print("\n4. Degradation heatmap:")
    plot_degradation_heatmap()
    print("\n5. Flip rate bar chart:")
    plot_flip_rate_bars()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

### Acceptance Criteria

- [ ] `scripts/generate_tables.py` runs end-to-end, producing 4 LaTeX tables in `experiments/tables/`
- [ ] `scripts/generate_figures.py` runs end-to-end, producing 5 figure pairs (PNG + PDF) in `experiments/figures/`
- [ ] All figures are 300 DPI, publication-quality with proper axis labels, legends, and titles
- [ ] LaTeX tables compile correctly when included in a LaTeX document
- [ ] Tables include mean +/- std formatting with proper LaTeX math mode
- [ ] Degradation heatmap uses a diverging colormap centered at 0
- [ ] Cumulative PnL plots show median with min-max bands across seeds
- [ ] Pipeline colors are consistent across all figures
- [ ] All scripts are deterministic (same input data produces identical output)

### Files to Create/Modify

- **Create:** `scripts/generate_tables.py`
- **Create:** `scripts/generate_figures.py`
- **Output:** `experiments/tables/benign_performance.tex`
- **Output:** `experiments/tables/attack_degradation.tex`
- **Output:** `experiments/tables/cgate_regimes.tex`
- **Output:** `experiments/tables/flip_rates.tex`
- **Output:** `experiments/figures/cumulative_pnl_benign.{png,pdf}`
- **Output:** `experiments/figures/cumulative_pnl_under_attack.{png,pdf}`
- **Output:** `experiments/figures/delta_violins.{png,pdf}`
- **Output:** `experiments/figures/degradation_heatmap.{png,pdf}`
- **Output:** `experiments/figures/flip_rate_bars.{png,pdf}`

### Dependencies

- P8-T2 (aggregated results CSVs, per-run JSON files)
- P8-T3 (C-Gate analysis: `regime_frequencies.csv`)
- `matplotlib`, `seaborn`, `pandas`

### Human Checkpoint

Before proceeding to P8-T5:
1. **Visual review all figures.** Are axes readable? Are legends clear? Do colors distinguish systems well?
2. **Verify table formatting.** Copy a `.tex` file into your thesis LaTeX project and compile. Does it render correctly?
3. **Sanity check the degradation heatmap.** Does Trinity (the robust system) show less degradation than single-agent baselines? This is the core thesis claim — if it doesn't hold, investigate before producing the final write-up.
4. **Check for visual artifacts.** Overlapping labels, clipped text, missing data points.

---

## P8-T5: Statistical Significance Testing

**Estimated time:** ~3 hours
**Dependencies:** P8-T2 complete (all per-run results with per-seed data)

### Context

With 5 seeds per condition, we have small samples (n=5). Standard parametric tests (t-tests) may not be reliable. We use non-parametric methods:
- **Wilcoxon signed-rank test:** Paired comparison between two systems across the same seeds and conditions. Tests whether one system consistently outperforms the other.
- **Bootstrap confidence intervals:** Resample with replacement to estimate the sampling distribution of mean metrics. Provides 95% CIs without distributional assumptions.
- **Cohen's d effect size:** Standardized difference between two groups. Indicates practical significance beyond just statistical significance.

These tests answer:
1. Is Trinity's robustness advantage over Executor-only statistically significant?
2. Is the C-Gate's contribution (Trinity vs. Trinity-no-CGate) significant?
3. Is channel independence important (Trinity vs. Fused-LLM-RL)?

### Objective

Perform rigorous statistical testing on all pairwise system comparisons, compute confidence intervals for all key metrics, and produce a summary that can be cited in the thesis.

### Detailed Instructions

1. **Create `scripts/statistical_tests.py`:**

```python
"""
Statistical significance testing for pairwise system comparisons.

Tests:
  1. Wilcoxon signed-rank test (non-parametric paired comparison)
  2. Bootstrap 95% confidence intervals for mean metrics
  3. Cohen's d effect size

Produces:
  - experiments/results/statistical_tests.json
  - experiments/results/significance_summary.csv

Usage:
    python scripts/statistical_tests.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from itertools import combinations

RESULTS_DIR = Path("experiments/results")
OUTPUT_DIR = RESULTS_DIR


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    statistic=np.mean,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (e.g. 0.95 for 95% CI)
        statistic: function to compute (default: np.mean)
        seed: random seed for reproducibility

    Returns:
        dict with "point_estimate", "ci_lower", "ci_upper", "ci_level"
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = np.percentile(boot_stats, alpha / 2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    return {
        "point_estimate": float(statistic(data)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size (paired or unpaired).
    Interpretation:
        |d| < 0.2  = negligible
        0.2-0.5    = small
        0.5-0.8    = medium
        > 0.8      = large
    """
    diff = group1 - group2
    return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-10))


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def wilcoxon_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Args:
        values_a, values_b: paired observations (same seeds, same conditions)
        alternative: "two-sided", "greater", or "less"

    Returns:
        dict with "statistic", "p_value", "significant_005", "significant_001"
    """
    diff = values_a - values_b
    # If all differences are zero, test is undefined
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_001": False,
            "n_pairs": len(diff),
            "warning": "all differences are zero",
        }

    # Wilcoxon requires at least 6 non-zero differences for a meaningful test.
    # With n=5 seeds, we may have very few pairs. Aggregate across conditions.
    non_zero = np.sum(diff != 0)
    if non_zero < 6:
        # With fewer than 6 non-zero pairs, report but flag as underpowered
        try:
            stat, p = wilcoxon(diff, alternative=alternative)
        except ValueError:
            return {
                "statistic": np.nan,
                "p_value": np.nan,
                "significant_005": False,
                "significant_001": False,
                "n_pairs": len(diff),
                "n_nonzero": int(non_zero),
                "warning": "too few non-zero differences for reliable test",
            }
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant_005": p < 0.05,
            "significant_001": p < 0.01,
            "n_pairs": len(diff),
            "n_nonzero": int(non_zero),
            "warning": "underpowered (< 6 non-zero differences)",
        }

    stat, p = wilcoxon(diff, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_005": p < 0.05,
        "significant_001": p < 0.01,
        "n_pairs": len(diff),
        "n_nonzero": int(non_zero),
    }


def run_pairwise_comparisons(df: pd.DataFrame) -> list[dict]:
    """
    For each pair of systems, compute:
    - Wilcoxon test on paired Sharpe ratios (aggregated across all attack conditions)
    - Cohen's d effect size
    - Bootstrap CIs for mean Sharpe difference

    Pairing is by (attack, seed) — same condition, same randomness.
    """
    pipelines = sorted(df["pipeline"].unique())
    results = []

    # Define the key comparisons (with hypotheses)
    key_comparisons = [
        ("Trinity", "Executor-only", "Trinity more robust than Executor-only"),
        ("Trinity", "Analyst-only", "Trinity more robust than Analyst-only"),
        ("Trinity", "Trinity-no-CGate", "C-Gate contributes to robustness"),
        ("Trinity", "Fused-LLM-RL", "Channel independence matters"),
    ]

    # Also do all pairwise for completeness
    all_pairs = list(combinations(pipelines, 2))

    for pipeline_a, pipeline_b, *hypothesis in [
        *key_comparisons,
        *[(a, b) for a, b in all_pairs
          if (a, b) not in [(k[0], k[1]) for k in key_comparisons]],
    ]:
        hyp = hypothesis[0] if hypothesis else f"{pipeline_a} vs. {pipeline_b}"

        if pipeline_a not in pipelines or pipeline_b not in pipelines:
            continue

        # Get paired observations: match on (attack, seed)
        df_a = df[df["pipeline"] == pipeline_a].set_index(["attack", "seed"])
        df_b = df[df["pipeline"] == pipeline_b].set_index(["attack", "seed"])

        # Inner join on shared (attack, seed) pairs
        shared_idx = df_a.index.intersection(df_b.index)
        if len(shared_idx) == 0:
            continue

        sharpe_a = df_a.loc[shared_idx, "sharpe"].values
        sharpe_b = df_b.loc[shared_idx, "sharpe"].values

        # Wilcoxon test (is A better than B?)
        wilcox = wilcoxon_test(sharpe_a, sharpe_b, alternative="greater")

        # Cohen's d
        d = cohens_d(sharpe_a, sharpe_b)

        # Bootstrap CI on Sharpe difference
        diff = sharpe_a - sharpe_b
        boot = bootstrap_ci(diff, n_bootstrap=10000, ci=0.95)

        result = {
            "comparison": f"{pipeline_a} vs. {pipeline_b}",
            "hypothesis": hyp,
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "n_paired_observations": len(shared_idx),
            "sharpe_a_mean": float(np.mean(sharpe_a)),
            "sharpe_b_mean": float(np.mean(sharpe_b)),
            "sharpe_diff_mean": float(np.mean(diff)),
            "wilcoxon": wilcox,
            "cohens_d": d,
            "effect_size_interpretation": interpret_effect_size(d),
            "bootstrap_ci_95": boot,
        }
        results.append(result)

    return results


def run_per_attack_tests(df: pd.DataFrame) -> list[dict]:
    """
    For the key comparison (Trinity vs. Executor-only), run Wilcoxon per attack.
    This shows whether the robustness advantage is consistent across attack types.
    """
    results = []
    attack_names = sorted(df["attack"].unique())

    for attack in attack_names:
        if attack == "benign":
            continue

        ad = df[df["attack"] == attack]
        trinity = ad[ad["pipeline"] == "Trinity"].sort_values("seed")
        executor = ad[ad["pipeline"] == "Executor-only"].sort_values("seed")

        if trinity.empty or executor.empty:
            continue

        # Match on seed
        common_seeds = set(trinity["seed"]) & set(executor["seed"])
        if len(common_seeds) < 3:
            continue

        t_sharpe = trinity[trinity["seed"].isin(common_seeds)].sort_values("seed")["sharpe"].values
        e_sharpe = executor[executor["seed"].isin(common_seeds)].sort_values("seed")["sharpe"].values

        d = cohens_d(t_sharpe, e_sharpe)
        diff = t_sharpe - e_sharpe
        boot = bootstrap_ci(diff, n_bootstrap=10000, ci=0.95)

        results.append({
            "attack": attack,
            "n_seeds": len(common_seeds),
            "trinity_sharpe_mean": float(np.mean(t_sharpe)),
            "executor_sharpe_mean": float(np.mean(e_sharpe)),
            "sharpe_diff_mean": float(np.mean(diff)),
            "cohens_d": d,
            "effect_size": interpret_effect_size(d),
            "bootstrap_ci_lower": boot["ci_lower"],
            "bootstrap_ci_upper": boot["ci_upper"],
        })

    return results


def main():
    print("Loading results...")
    df = pd.read_csv(RESULTS_DIR / "summary_all_runs.csv")
    print(f"  {len(df)} runs, {df['pipeline'].nunique()} pipelines, "
          f"{df['attack'].nunique()} conditions")

    print("\nRunning pairwise comparisons (aggregated across attacks)...")
    pairwise = run_pairwise_comparisons(df)

    print("\nRunning per-attack tests (Trinity vs. Executor-only)...")
    per_attack = run_per_attack_tests(df)

    # Compile full results
    full_results = {
        "pairwise_comparisons": pairwise,
        "per_attack_trinity_vs_executor": per_attack,
        "metadata": {
            "n_total_runs": len(df),
            "pipelines": df["pipeline"].unique().tolist(),
            "attacks": df["attack"].unique().tolist(),
            "n_seeds": df["seed"].nunique(),
            "tests_used": [
                "Wilcoxon signed-rank (non-parametric, paired)",
                "Bootstrap 95% CI (10000 resamples)",
                "Cohen's d effect size",
            ],
        },
    }

    # Save full results
    with open(OUTPUT_DIR / "statistical_tests.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nSaved full results to {OUTPUT_DIR / 'statistical_tests.json'}")

    # Generate summary CSV
    summary_rows = []
    for comp in pairwise:
        summary_rows.append({
            "Comparison": comp["comparison"],
            "Hypothesis": comp["hypothesis"],
            "N": comp["n_paired_observations"],
            "Sharpe_A": f"{comp['sharpe_a_mean']:.4f}",
            "Sharpe_B": f"{comp['sharpe_b_mean']:.4f}",
            "Diff": f"{comp['sharpe_diff_mean']:.4f}",
            "p_value": f"{comp['wilcoxon']['p_value']:.4f}"
                        if not np.isnan(comp['wilcoxon'].get('p_value', np.nan))
                        else "N/A",
            "Sig_005": comp["wilcoxon"]["significant_005"],
            "Cohens_d": f"{comp['cohens_d']:.3f}",
            "Effect": comp["effect_size_interpretation"],
            "CI_95_lower": f"{comp['bootstrap_ci_95']['ci_lower']:.4f}",
            "CI_95_upper": f"{comp['bootstrap_ci_95']['ci_upper']:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "significance_summary.csv", index=False)
    print(f"Saved summary to {OUTPUT_DIR / 'significance_summary.csv'}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    for comp in pairwise:
        sig = "***" if comp["wilcoxon"]["significant_001"] else \
              "**" if comp["wilcoxon"]["significant_005"] else "n.s."
        print(f"\n{comp['comparison']} ({comp['hypothesis']}):")
        print(f"  Sharpe diff: {comp['sharpe_diff_mean']:+.4f} "
              f"[{comp['bootstrap_ci_95']['ci_lower']:+.4f}, "
              f"{comp['bootstrap_ci_95']['ci_upper']:+.4f}]")
        print(f"  Wilcoxon p={comp['wilcoxon']['p_value']:.4f} {sig}")
        print(f"  Cohen's d={comp['cohens_d']:.3f} ({comp['effect_size_interpretation']})")


if __name__ == "__main__":
    main()
```

### Acceptance Criteria

- [ ] `scripts/statistical_tests.py` runs end-to-end without errors
- [ ] `experiments/results/statistical_tests.json` contains all pairwise comparisons with Wilcoxon, bootstrap CI, and Cohen's d
- [ ] `experiments/results/significance_summary.csv` contains a human-readable summary table
- [ ] Wilcoxon test correctly handles small sample sizes (n=5 seeds) with appropriate warnings
- [ ] Bootstrap CIs use 10,000 resamples with fixed seed for reproducibility
- [ ] Cohen's d is computed and interpreted with standard thresholds (negligible/small/medium/large)
- [ ] Per-attack breakdown is computed for the key Trinity vs. Executor-only comparison
- [ ] The key thesis questions are answered:
  - Is Trinity significantly more robust than Executor-only? (p < 0.05)
  - Does the C-Gate contribute significantly? (Trinity vs. Trinity-no-CGate)
  - Does channel independence matter? (Trinity vs. Fused-LLM-RL)
- [ ] Warnings are emitted when sample sizes are too small for reliable inference

### Files to Create/Modify

- **Create:** `scripts/statistical_tests.py`
- **Output:** `experiments/results/statistical_tests.json`
- **Output:** `experiments/results/significance_summary.csv`

### Dependencies

- P8-T2 (`experiments/results/summary_all_runs.csv`)
- `scipy.stats` for Wilcoxon test
- `numpy` for bootstrap

### Human Checkpoint

After running statistical tests:
1. **Check p-values.** Are any key comparisons non-significant (p > 0.05)? If Trinity vs. Executor-only is not significant, the thesis claim is weakened — consider increasing the number of seeds or analyzing why.
2. **Check effect sizes.** Even if p < 0.05, a negligible Cohen's d means the practical difference is small. Report both.
3. **Review bootstrap CIs.** Do any CIs include zero? If the CI for Trinity - Executor-only Sharpe difference includes zero, the advantage is not robust.
4. **Per-attack consistency.** Is Trinity's advantage consistent across all attack types, or only for certain ones? This nuance matters for the thesis claims.
5. **Power analysis.** With only 5 seeds, the Wilcoxon test has limited power. If key results are non-significant, note the power limitation and consider running additional seeds.
