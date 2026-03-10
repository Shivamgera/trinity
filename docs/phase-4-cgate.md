# Phase 4: Consistency Gate — C-Gate

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Project Root:** /Users/shivamgera/projects/research1
**Timeline:** Week 7 (~8–10 hours total across 2 tasks)

---

## P4-T1: Implement C-Gate Divergence Computation and Core Logic

**Estimated time:** ~3 hours
**Dependencies:** Phase 3 complete (Analyst schema, conversion, precomputed signals all working)

### Context

The Consistency Gate (C-Gate) is the central arbitration mechanism in the Robust Trinity architecture. It takes the Analyst's discrete decision `d_LLM` (an index in {0,1,2} for {flat,long,short}) and the Executor's full policy distribution `π_RL`, and computes their divergence using `Δ = 1 - π_RL(d_LLM | s_t)`. Based on the divergence value `Δ`, it assigns a regime that determines how the system acts.

**Key design facts:**
- Action space: {flat=0, long=1, short=2}
- `d_LLM`: discrete decision from the Analyst (the LLM outputs `(d, r)` — decision and reasoning only, no confidence score)
- `π_RL`: policy distribution from `get_policy_distribution()` in `src/executor/policy.py`
- Δ = 1 - π_RL[d_LLM], bounded in [0, 1]: 0 when RL assigns probability 1 to LLM's action, ~1 when RL assigns near-zero probability
- No scipy needed, no pseudo-distribution needed
- Three regimes:
  - **Agreement** (Δ ≤ τ_low=0.1): Both channels agree. Execute `argmax(π_RL)` normally.
  - **Ambiguity** (τ_low < Δ ≤ τ_high=0.4): Moderate disagreement. Execute `argmax(π_RL)` in conservative mode (50% position, 2% stop). Guardian activates conservative mode.
  - **Conflict** (Δ > τ_high=0.4): Strong disagreement. Fail-safe to flat (`action=0`). Analyst reasoning trace forwarded to Guardian for logging.
- Initial thresholds τ_low=0.1, τ_high=0.4 will be calibrated later via ROC analysis (not in this task).
- The reasoning trace `r` from the Analyst is NOT used in Δ computation. It is forwarded to the Guardian only during conflict regime for audit/inspection.

**Already available:**
- `src/analyst/schema.py` — `TradeSignal` Pydantic model
- `src/executor/policy.py` — `get_policy_distribution(model, obs, vec_normalize=...) → np.array(3,)` and `load_executor(model_dir) → (PPO, VecNormalize)`

### Objective

Implement the divergence computation module and the C-Gate class with regime classification and action logic.

### Detailed Instructions

#### Step 1: Create `src/cgate/__init__.py`

Empty file to make `src/cgate` a Python package.

#### Step 2: Create `src/cgate/divergence.py`

```python
"""Divergence computation for the Consistency Gate.

Computes Δ = 1 - π_RL(d_LLM | s_t), measuring how much probability
the Executor's policy assigns to the Analyst's chosen action.

Δ ∈ [0, 1]:
  - Δ ≈ 0: RL strongly agrees with LLM's action (high probability)
  - Δ ≈ 1: RL assigns near-zero probability to LLM's action (strong disagreement)
"""

from __future__ import annotations

import numpy as np


# Action mapping: LLM decision string → action index
ACTION_MAP: dict[str, int] = {
    "hold": 0,
    "flat": 0,
    "buy": 1,
    "long": 1,
    "sell": 2,
    "short": 2,
}


def compute_delta(d_llm: str | int, pi_rl: np.ndarray) -> float:
    """Compute C-Gate divergence Δ = 1 - π_RL(d_LLM).

    Args:
        d_llm: Analyst's decision — either a string ("hold"/"buy"/"sell")
               or an integer action index (0/1/2).
        pi_rl: Executor's policy distribution, shape (3,), sums to 1.

    Returns:
        Δ value in [0, 1].

    Raises:
        ValueError: If d_llm is not a valid action or pi_rl is invalid.
    """
    # Validate pi_rl
    pi_rl = np.asarray(pi_rl, dtype=np.float64)
    if pi_rl.shape != (3,):
        raise ValueError(f"pi_rl must have shape (3,), got {pi_rl.shape}")
    if not np.isclose(pi_rl.sum(), 1.0, atol=1e-6):
        raise ValueError(f"pi_rl must sum to 1.0, got {pi_rl.sum():.6f}")
    if np.any(pi_rl < 0):
        raise ValueError(f"pi_rl contains negative values: {pi_rl}")

    # Resolve action index
    if isinstance(d_llm, str):
        d_llm_lower = d_llm.lower().strip()
        if d_llm_lower not in ACTION_MAP:
            raise ValueError(
                f"Unknown decision '{d_llm}'. Must be one of: {list(ACTION_MAP.keys())}"
            )
        action_idx = ACTION_MAP[d_llm_lower]
    elif isinstance(d_llm, (int, np.integer)):
        if d_llm not in (0, 1, 2):
            raise ValueError(f"Action index must be 0, 1, or 2, got {d_llm}")
        action_idx = int(d_llm)
    else:
        raise TypeError(f"d_llm must be str or int, got {type(d_llm)}")

    delta = 1.0 - float(pi_rl[action_idx])
    return float(np.clip(delta, 0.0, 1.0))
```

#### Step 3: Create `src/cgate/gate.py`

```python
"""Consistency Gate — core arbitration logic for the Robust Trinity."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.cgate.divergence import compute_delta


@dataclass
class CGateResult:
    """Output of the Consistency Gate evaluation."""
    delta: float                                           # Divergence Δ = 1 - π_RL(d_LLM)
    regime: Literal["agreement", "ambiguity", "conflict"]  # classified regime
    action: int                                            # final action: 0=flat, 1=long, 2=short


class ConsistencyGate:
    """Consistency Gate — core coordination mechanism of the Robust Trinity.

    Compares the Analyst's discrete decision d_LLM against the Executor's
    policy distribution π_RL using Δ = 1 - π_RL(d_LLM).

    Three regimes:
      - Agreement (Δ ≤ τ_low):  Execute argmax(π_RL) normally
      - Ambiguity (τ_low < Δ ≤ τ_high): Execute argmax(π_RL) in conservative mode
      - Conflict  (Δ > τ_high): Fail-safe to flat (action=0)
    """

    def __init__(self, tau_low: float = 0.1, tau_high: float = 0.4) -> None:
        if not 0 <= tau_low <= tau_high <= 1:
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ τ_low ≤ τ_high ≤ 1, "
                f"got τ_low={tau_low}, τ_high={tau_high}"
            )
        self.tau_low = tau_low
        self.tau_high = tau_high

    def evaluate(self, d_llm: str | int, pi_rl: np.ndarray) -> CGateResult:
        """Evaluate the C-Gate for a single timestep.

        Args:
            d_llm: Analyst's decision (string or action index).
            pi_rl: Executor's policy distribution, shape (3,).

        Returns:
            CGateResult with delta, regime, and recommended action.
        """
        delta = compute_delta(d_llm, pi_rl)
        rl_action = int(np.argmax(pi_rl))

        if delta <= self.tau_low:
            regime = "agreement"
            action = rl_action
        elif delta <= self.tau_high:
            regime = "ambiguity"
            action = rl_action  # Same action, but Guardian applies conservative mode
        else:
            regime = "conflict"
            action = 0  # Fail-safe to flat

        return CGateResult(delta=delta, regime=regime, action=action)
```

#### Step 4: Create `tests/test_cgate.py`

```python
"""Comprehensive tests for the Consistency Gate."""
import numpy as np
import pytest
from src.cgate.divergence import compute_delta
from src.cgate.gate import ConsistencyGate, CGateResult


class TestComputeDelta:
    """Tests for the compute_delta function."""

    def test_perfect_agreement(self):
        """When RL assigns probability 1.0 to LLM's action, Δ = 0."""
        delta = compute_delta("buy", np.array([0.0, 1.0, 0.0]))
        assert delta == 0.0

    def test_perfect_disagreement(self):
        """When RL assigns probability 0.0 to LLM's action, Δ = 1."""
        delta = compute_delta("buy", np.array([0.5, 0.0, 0.5]))
        assert delta == 1.0

    def test_moderate_agreement(self):
        """When RL assigns moderate probability, Δ is moderate."""
        delta = compute_delta("buy", np.array([0.2, 0.6, 0.2]))
        assert abs(delta - 0.4) < 1e-10

    def test_uniform_distribution(self):
        """Uniform π_RL → Δ = 1 - 1/3 ≈ 0.667."""
        delta = compute_delta("hold", np.array([1/3, 1/3, 1/3]))
        assert abs(delta - 2/3) < 1e-10

    def test_bounded_zero_one(self):
        """Δ should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            delta = compute_delta(d_llm, pi_rl)
            assert 0.0 <= delta <= 1.0 + 1e-10, f"Δ={delta} out of bounds"

    def test_string_and_int_equivalent(self):
        """String 'buy' and int 1 should give the same result."""
        pi_rl = np.array([0.2, 0.5, 0.3])
        assert compute_delta("buy", pi_rl) == compute_delta(1, pi_rl)
        assert compute_delta("hold", pi_rl) == compute_delta(0, pi_rl)
        assert compute_delta("sell", pi_rl) == compute_delta(2, pi_rl)

    def test_alternative_names(self):
        """'flat'/'long'/'short' should map to 0/1/2."""
        pi_rl = np.array([0.3, 0.5, 0.2])
        assert compute_delta("flat", pi_rl) == compute_delta("hold", pi_rl)
        assert compute_delta("long", pi_rl) == compute_delta("buy", pi_rl)
        assert compute_delta("short", pi_rl) == compute_delta("sell", pi_rl)

    def test_invalid_decision_raises(self):
        """Unknown decision string should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            compute_delta("unknown", np.array([0.3, 0.4, 0.3]))

    def test_invalid_pi_rl_shape(self):
        """Wrong shape pi_rl should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            compute_delta("buy", np.array([0.5, 0.5]))

    def test_invalid_pi_rl_sum(self):
        """pi_rl not summing to 1 should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            compute_delta("buy", np.array([0.5, 0.5, 0.5]))


class TestConsistencyGate:
    """Tests for the C-Gate regime classification and action selection."""

    @pytest.fixture
    def gate(self):
        return ConsistencyGate(tau_low=0.1, tau_high=0.4)

    def test_agreement_regime(self, gate):
        """When RL strongly agrees with LLM → agreement, execute argmax(π_RL)."""
        # RL assigns 0.95 to "long" (action 1), LLM says "long"
        pi_rl = np.array([0.025, 0.95, 0.025])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "agreement"
        assert result.delta == pytest.approx(0.05, abs=1e-10)
        assert result.action == 1  # argmax of pi_rl

    def test_ambiguity_regime(self, gate):
        """When RL moderately disagrees with LLM → ambiguity."""
        # RL assigns 0.7 to "hold" (action 0), LLM says "long" (action 1)
        pi_rl = np.array([0.7, 0.2, 0.1])
        result = gate.evaluate("long", pi_rl)
        # Δ = 1 - 0.2 = 0.8 → conflict actually
        # Let's pick values that give ambiguity: Δ = 1 - 0.75 = 0.25
        pi_rl = np.array([0.1, 0.75, 0.15])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "ambiguity"
        assert result.delta == pytest.approx(0.25, abs=1e-10)
        assert result.action == 1  # argmax of pi_rl (conservative mode applied by Guardian)

    def test_conflict_regime(self, gate):
        """When RL strongly disagrees with LLM → conflict, fail-safe to flat."""
        # RL assigns 0.05 to "long" (action 1), LLM says "long"
        pi_rl = np.array([0.6, 0.05, 0.35])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "conflict"
        assert result.delta == pytest.approx(0.95, abs=1e-10)
        assert result.action == 0  # flat (safe default)

    def test_conflict_forces_flat(self, gate):
        """In conflict regime, action must always be 0 (flat)."""
        # RL assigns 0.0 to "sell" (action 2)
        pi_rl = np.array([0.5, 0.5, 0.0])
        result = gate.evaluate("sell", pi_rl)
        assert result.regime == "conflict"
        assert result.action == 0

    def test_delta_always_bounded(self, gate):
        """Δ should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)
            assert 0.0 <= result.delta <= 1.0 + 1e-10

    def test_custom_thresholds(self):
        """Gate respects custom threshold values."""
        gate = ConsistencyGate(tau_low=0.01, tau_high=0.05)
        # Δ = 1 - 0.5 = 0.5, which > 0.05 → conflict
        pi_rl = np.array([0.3, 0.5, 0.2])
        result = gate.evaluate("buy", pi_rl)
        assert result.regime == "conflict"

    def test_invalid_thresholds(self):
        """Invalid thresholds should raise ValueError."""
        with pytest.raises(ValueError):
            ConsistencyGate(tau_low=0.5, tau_high=0.3)  # low > high
        with pytest.raises(ValueError):
            ConsistencyGate(tau_low=-0.1, tau_high=0.5)  # negative
```

### Acceptance Criteria

- [ ] Δ values are always in [0, 1] for any valid action and distribution
- [ ] Δ = 0 when π_RL assigns probability 1.0 to d_LLM, Δ = 1 when probability 0.0
- [ ] String and integer action inputs produce identical results
- [ ] C-Gate correctly classifies agreement, ambiguity, and conflict regimes
- [ ] Agreement: `action = argmax(π_RL)`
- [ ] Ambiguity: `action = argmax(π_RL)` (conservative mode applied by Guardian)
- [ ] Conflict: `action = 0` (flat), reasoning forwarded to Guardian
- [ ] All tests pass: `pytest tests/test_cgate.py -v`

### Files to Create/Modify

- `src/cgate/__init__.py` (create)
- `src/cgate/divergence.py` (create)
- `src/cgate/gate.py` (create)
- `tests/test_cgate.py` (create)

### Dependencies

- Phase 3 complete (need `src/analyst/schema.py` for type compatibility)

### Human Checkpoint

- Run `pytest tests/test_cgate.py -v` and verify all tests pass.
- Manually verify: `compute_delta("long", np.array([0.1, 0.1, 0.8]))` should return 0.9, which puts it in the conflict regime (> 0.4). If not, the thresholds may need future calibration, but the math should be correct.

---

## P4-T2: C-Gate Integration Test with Real Agent Outputs

**Estimated time:** ~2.5 hours
**Dependencies:** P4-T1 must be complete; Phase 2 (Executor) and Phase 3 (Analyst precomputed signals) must be complete.

### Context

P4-T1 built the C-Gate core logic and unit-tested it with synthetic inputs. Now we need to run the C-Gate on **real agent outputs** — the frozen Executor's policy distributions and the Analyst's precomputed decisions — to verify that the system behaves sensibly under benign (non-adversarial) conditions.

This integration test serves two critical purposes:
1. **Validation**: Ensure the full pipeline (features → Executor → π_RL, headline → Analyst → d_LLM, C-Gate → regime) works end-to-end.
2. **Baseline characterization**: The distribution of Δ values under benign conditions establishes the baseline that adversarial experiments (Phase 8) will later perturb. If the system is already in conflict 50%+ of the time without any attack, something is fundamentally wrong.

**Critical design note — env-rollout approach:**

The C-Gate integration **must** run the Executor through `TradingEnv` as a full trading simulation, not by looking up single feature rows from the parquet file. This is essential because:

- The Executor expects a 423-dimensional observation (30-day lookback × 14 features + 3 portfolio state dims). A single parquet row is only 14 features — feeding it directly would crash or produce garbage.
- Portfolio state (position, unrealized PnL, time since trade) is maintained by the environment and evolves with each action. There is no way to reconstruct it from a static feature file.
- `VecNormalize` statistics from training must be applied to observations. The `load_executor()` function returns both the model and the `VecNormalize` wrapper; both must be used.

The approach:
1. Load the frozen Executor via `load_executor()` → `(model, vec_normalize)`
2. Create `TradingEnv` on the test split with `random_start=False`
3. At each step, read `info["date"]` (added to `TradingEnv._get_info()` in P2-T1) to find the calendar date
4. Look up `date_to_signal[date]` for the Analyst's precomputed decision
5. Get `π_RL` via `get_policy_distribution(model, obs, vec_normalize=vec_norm)`
6. Feed both into `gate.evaluate(d_llm, pi_rl)` to get the C-Gate result
7. Execute the C-Gate's recommended action (not the Executor's raw action) in the environment

On days where no precomputed Analyst signal exists (e.g., a trading day with no headline), the system treats this as a **conflict** (Δ = 1.0) and goes flat. This is the conservative default — the Analyst channel is silent, so the system cannot verify agreement.

**Already available:**
- `src/cgate/divergence.py` — `compute_delta()` function
- `src/cgate/gate.py` — `ConsistencyGate` class
- `src/analyst/schema.py` — `TradeSignal`
- `data/processed/precomputed_signals.json` — precomputed Analyst signals from P3-T2
- `experiments/executor/best_model/model.zip` + `vec_normalize.pkl` — frozen PPO model
- `data/processed/aapl_features.parquet` — 14 z-normalized features, indexed by date
- `src/executor/policy.py` — `get_policy_distribution(model, obs, vec_normalize=...)` and `load_executor()`
- `src/executor/env.py` — `TradingEnv` with `current_date` property and `info["date"]`

### Objective

Run the C-Gate through a full TradingEnv simulation on the test split, log Δ statistics, and verify benign-condition behavior.

### Detailed Instructions

#### Step 1: Create `scripts/cgate_integration.py`

```python
"""
C-Gate integration test: Run full Trinity simulation on the test split.

This script runs the Executor through TradingEnv while synchronizing
with precomputed Analyst signals via date matching.  At each step the
C-Gate arbitrates between the two channels and the recommended action
is executed in the environment, producing a realistic trading trajectory.

Pipeline per step:
  1. env.step() → obs, info["date"]
  2. date → date_to_signal[date] → d_LLM
  3. obs → get_policy_distribution(model, obs, vec_normalize) → π_RL
  4. gate.evaluate(d_LLM, π_RL) → CGateResult(delta, regime, action)
  5. Execute CGateResult.action in the env on the *next* step

Usage:
    python scripts/cgate_integration.py
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    from src.cgate.gate import ConsistencyGate
    from src.executor.policy import get_policy_distribution, load_executor
    from src.executor.env_factory import create_vec_env

    # ---- Load Executor model + VecNormalize ----
    model_dir = "experiments/executor/best_model"
    logger.info(f"Loading Executor from {model_dir}")
    model, vec_norm = load_executor(model_dir)

    # ---- Load precomputed Analyst signals ----
    signals_path = "data/processed/precomputed_signals.json"
    logger.info(f"Loading precomputed signals from {signals_path}")
    with open(signals_path, "r") as f:
        signals_cache = json.load(f)

    from src.analyst.schema import TradeSignal

    date_to_signal: dict[str, TradeSignal] = {}
    for key, entry in signals_cache.items():
        date = entry["date"]
        signal = TradeSignal(
            reasoning=entry["reasoning"],
            decision=entry["decision"],
        )
        date_to_signal[date] = signal

    logger.info(f"Loaded {len(date_to_signal)} Analyst signals.")

    # ---- Create TradingEnv on the test split ----
    # random_start=False ensures we start at the earliest possible step
    # (after the lookback window) and walk through the full test period.
    env_fns = create_vec_env(
        n_envs=1,
        split="test",
        random_start=False,
        seed=42,
    )
    raw_env = DummyVecEnv(env_fns)

    # Wrap in VecNormalize with frozen stats from training
    if vec_norm is not None:
        eval_vec = VecNormalize(raw_env, norm_obs=True, norm_reward=False)
        eval_vec.obs_rms = vec_norm.obs_rms
        eval_vec.ret_rms = vec_norm.ret_rms
        eval_vec.training = False
    else:
        eval_vec = raw_env

    # ---- Run C-Gate integration loop ----
    gate = ConsistencyGate(tau_low=0.1, tau_high=0.4)

    obs = eval_vec.reset()
    done = False
    results = []
    missing_signal_dates = []

    while not done:
        # Extract the underlying env's info to get the current date.
        # DummyVecEnv wraps a single env — access it directly.
        inner_env = raw_env.envs[0]
        current_date = inner_env.current_date

        # --- Get Executor's policy distribution ---
        # obs from VecNormalize is already normalized; however
        # get_policy_distribution expects the RAW obs and normalizes
        # internally.  We pass the raw observation and vec_normalize.
        raw_obs = raw_env.get_attr("_get_observation")[0]()  # unnormalized
        pi_rl = get_policy_distribution(model, raw_obs, vec_normalize=vec_norm)

        # --- Get Analyst decision for this date ---
        if current_date and current_date in date_to_signal:
            signal = date_to_signal[current_date]
            d_llm = signal.decision
            has_signal = True
        else:
            # No Analyst signal for this date → treat as conflict (Δ=1.0)
            d_llm = None
            has_signal = False
            if current_date:
                missing_signal_dates.append(current_date)

        # --- Run C-Gate ---
        if has_signal:
            cgate_result = gate.evaluate(d_llm, pi_rl)
        else:
            # Missing signal → forced conflict, go flat
            from src.cgate.gate import CGateResult
            cgate_result = CGateResult(delta=1.0, regime="conflict", action=0)

        # --- Execute the C-Gate's action in the env ---
        # We override the Executor's raw action with the C-Gate's decision.
        action = np.array([cgate_result.action])
        obs, reward, done_arr, info = eval_vec.step(action)
        done = done_arr[0]

        results.append({
            "date": current_date,
            "delta": cgate_result.delta,
            "regime": cgate_result.regime,
            "action": cgate_result.action,
            "analyst_decision": d_llm if has_signal else "MISSING",
            "has_analyst_signal": has_signal,
            "pi_rl": pi_rl.tolist(),
            "reward": float(reward[0]),
            "portfolio_return": info[0].get("portfolio_return", 0.0),
            "position": info[0].get("position", 0.0),
            "price": info[0].get("price", 0.0),
        })

    eval_vec.close()

    logger.info(f"Successfully processed {len(results)} timesteps.")
    logger.info(f"Missing Analyst signals on {len(missing_signal_dates)} dates.")

    # ---- Compute statistics ----
    deltas = np.array([r["delta"] for r in results])
    regimes = Counter(r["regime"] for r in results)
    n = len(results)

    # Trading performance
    port_returns = np.array([r["portfolio_return"] for r in results])
    cumulative = np.cumprod(1 + port_returns)
    total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0

    from src.executor.evaluate import compute_sharpe_ratio, compute_max_drawdown
    sharpe = compute_sharpe_ratio(port_returns)
    max_dd = compute_max_drawdown(cumulative)

    logger.info(f"\n{'='*60}")
    logger.info(f"C-GATE INTEGRATION RESULTS ({n} timesteps)")
    logger.info(f"{'='*60}")
    logger.info(f"Δ statistics:")
    logger.info(f"  Mean:   {deltas.mean():.4f}")
    logger.info(f"  Median: {np.median(deltas):.4f}")
    logger.info(f"  Std:    {deltas.std():.4f}")
    logger.info(f"  Min:    {deltas.min():.4f}")
    logger.info(f"  Max:    {deltas.max():.4f}")
    logger.info(f"  P25:    {np.percentile(deltas, 25):.4f}")
    logger.info(f"  P75:    {np.percentile(deltas, 75):.4f}")
    logger.info(f"\nRegime distribution:")
    for regime in ["agreement", "ambiguity", "conflict"]:
        count = regimes.get(regime, 0)
        pct = count / n * 100 if n > 0 else 0
        logger.info(f"  {regime:12s}: {count:4d} ({pct:5.1f}%)")
    logger.info(f"\nTrading performance (C-Gate controlled):")
    logger.info(f"  Sharpe ratio:  {sharpe:.4f}")
    logger.info(f"  Total return:  {total_return:.4f}")
    logger.info(f"  Max drawdown:  {max_dd:.4f}")

    # ---- Print examples for each regime ----
    logger.info(f"\nExample outputs per regime:")
    for regime in ["agreement", "ambiguity", "conflict"]:
        examples = [r for r in results if r["regime"] == regime]
        if examples:
            ex = examples[0]
            logger.info(f"\n  [{regime.upper()}] Date: {ex['date']}")
            logger.info(f"    Δ = {ex['delta']:.4f}")
            logger.info(f"    d_LLM = {ex['analyst_decision']}")
            logger.info(f"    π_RL   = {ex['pi_rl']}")
            logger.info(f"    Action: {ex['action']} (0=flat, 1=long, 2=short)")

    # ---- Log to W&B ----
    try:
        import wandb

        wandb.init(project="robust-trinity", name="cgate-integration", reinit=True)

        # Summary statistics
        wandb.log({
            "cgate/delta_mean": deltas.mean(),
            "cgate/delta_median": float(np.median(deltas)),
            "cgate/delta_std": deltas.std(),
            "cgate/delta_min": deltas.min(),
            "cgate/delta_max": deltas.max(),
            "cgate/pct_agreement": regimes.get("agreement", 0) / n,
            "cgate/pct_ambiguity": regimes.get("ambiguity", 0) / n,
            "cgate/pct_conflict": regimes.get("conflict", 0) / n,
            "cgate/n_timesteps": n,
            "cgate/n_missing_signals": len(missing_signal_dates),
            "cgate/sharpe_ratio": sharpe,
            "cgate/total_return": total_return,
            "cgate/max_drawdown": max_dd,
        })

        # Δ histogram
        wandb.log({"cgate/delta_histogram": wandb.Histogram(deltas, num_bins=50)})

        # Time series of Δ
        for i, r in enumerate(results):
            wandb.log({
                "cgate/delta_timeseries": r["delta"],
                "cgate/timestep": i,
            })

        # Full results table
        columns = ["date", "delta", "regime", "action",
                    "analyst_decision", "has_analyst_signal",
                    "reward", "portfolio_return", "position", "price"]
        table_data = []
        for r in results:
            row = [r[c] for c in columns]
            table_data.append(row)
        wandb.log({"cgate/results_table": wandb.Table(columns=columns, data=table_data)})

        wandb.finish()
        logger.info("Results logged to W&B.")
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")

    # ---- Save results locally ----
    output_path = "experiments/cgate/integration_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "statistics": {
                "n_timesteps": n,
                "delta_mean": float(deltas.mean()),
                "delta_median": float(np.median(deltas)),
                "delta_std": float(deltas.std()),
                "regimes": dict(regimes),
                "n_missing_signals": len(missing_signal_dates),
                "sharpe_ratio": sharpe,
                "total_return": total_return,
                "max_drawdown": max_dd,
            },
            "results": results,
        }, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # ---- Sanity checks ----
    assert len(results) > 0, "No results produced!"
    assert all(0 <= r["delta"] <= 1.0 + 1e-10 for r in results), "Δ out of bounds!"

    # Warn if too much conflict
    conflict_pct = regimes.get("conflict", 0) / n
    if conflict_pct > 0.5:
        logger.warning(
            f"⚠ {conflict_pct:.0%} of timesteps are in CONFLICT regime under benign conditions. "
            f"This suggests Analyst and Executor are fundamentally misaligned. "
            f"Investigate Analyst signals or consider adjusting thresholds."
        )

    # Verify all three regimes occur (soft check — warn, don't fail)
    for regime in ["agreement", "ambiguity", "conflict"]:
        if regime not in regimes:
            logger.warning(
                f"⚠ Regime '{regime}' never occurred. "
                f"Consider adjusting thresholds (tau_low={gate.tau_low}, tau_high={gate.tau_high})."
            )

    logger.info("\nIntegration test complete.")


if __name__ == "__main__":
    main()
```

#### Step 2: Create `tests/test_cgate_integration.py`

A lighter-weight test that can run in CI without the full model:

```python
"""Integration-style tests for C-Gate with synthetic but realistic inputs."""
import numpy as np
import pytest
from src.cgate.gate import ConsistencyGate


class TestCGateIntegrationSynthetic:
    """Test C-Gate with synthetic distributions that mimic real agent outputs."""

    @pytest.fixture
    def gate(self):
        return ConsistencyGate(tau_low=0.1, tau_high=0.4)

    def test_full_sequence(self, gate):
        """Simulate a sequence of timesteps with varying agreement levels."""
        rng = np.random.default_rng(42)
        regimes_seen = set()

        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)

            assert 0 <= result.delta <= 1.0 + 1e-10
            assert result.regime in ("agreement", "ambiguity", "conflict")
            assert result.action in (0, 1, 2)
            regimes_seen.add(result.regime)

        # With random distributions, all regimes should appear
        assert len(regimes_seen) == 3, f"Only saw regimes: {regimes_seen}"

    def test_agreement_sequence(self, gate):
        """When both agents consistently agree, should be mostly agreement."""
        regimes = []
        for _ in range(50):
            # RL strongly favors "long" (action 1), LLM says "long"
            pi_rl = np.array([0.02, 0.96, 0.02])
            result = gate.evaluate("long", pi_rl)
            regimes.append(result.regime)

        agreement_pct = regimes.count("agreement") / len(regimes)
        assert agreement_pct > 0.8, f"Expected >80% agreement, got {agreement_pct:.0%}"

    def test_conflict_always_flat(self, gate):
        """Every conflict-regime result must have action=0."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)
            if result.regime == "conflict":
                assert result.action == 0, (
                    f"Conflict regime should force flat, got action {result.action}"
                )
```

### Acceptance Criteria

- [ ] `scripts/cgate_integration.py` runs to completion without errors
- [ ] The script runs the Executor through `TradingEnv` (not by looking up single feature rows) — observations are 423-dimensional
- [ ] `VecNormalize` statistics are loaded from `experiments/executor/best_model/vec_normalize.pkl` via `load_executor()` and applied during inference
- [ ] Observations are the raw 423-dim vectors from the env, passed to `get_policy_distribution()` together with `vec_normalize`
- [ ] On dates with no precomputed Analyst signal, the system treats the step as conflict (Δ = 1.0) and goes flat
- [ ] All Δ values are bounded in [0, 1]
- [ ] All three regimes (agreement, ambiguity, conflict) occur at least once in the results
- [ ] Trading performance metrics (Sharpe, total return, max drawdown) are computed and logged
- [ ] If >50% of timesteps are conflict, a warning is emitted (but the test still passes — this is a signal for the human to investigate)
- [ ] Results are saved to `experiments/cgate/integration_results.json`
- [ ] Results are logged to W&B under project "robust-trinity"
- [ ] `pytest tests/test_cgate_integration.py -v` passes

### Files to Create/Modify

- `scripts/cgate_integration.py` (create)
- `tests/test_cgate_integration.py` (create)
- `experiments/cgate/integration_results.json` (auto-generated output)

### Dependencies

- P4-T1 must be complete (C-Gate core logic)
- Phase 3 must be complete (`data/processed/precomputed_signals.json`)
- Phase 2 must be complete (`experiments/executor/best_model/model.zip` + `vec_normalize.pkl`)
- P2-T1 must include the date-tracking additions (`TradingEnv.current_date` property, `info["date"]`)

### Human Checkpoint

- **Verify env-rollout approach**: Confirm the script uses `TradingEnv` with `random_start=False`, not direct parquet row lookups. The observation dimension must be 423, not 14.
- **Verify VecNormalize is loaded**: Check that `load_executor()` is used (not bare `PPO.load()`), and `vec_normalize` is passed to `get_policy_distribution()`.
- **Review the Δ distribution**: Under benign conditions, the majority of timesteps should be in agreement or ambiguity. If conflict is dominant, investigate whether:
  1. The Analyst is producing random/uncalibrated signals (check precomputed signals)
  2. The Executor's policy is degenerate (always outputs the same action)
  3. The date matching between `info["date"]` and `date_to_signal` is incorrect (date format mismatch)
- **Review missing signal handling**: Check how many dates have no Analyst signal. If it is the majority, the precomputed signals may have a date format mismatch or insufficient coverage.
- **Review example outputs**: For each regime, check that the printed d_LLM and π_RL make sense. In agreement examples, π_RL should assign high probability to d_LLM's action. In conflict examples, π_RL should assign near-zero probability to d_LLM's action.
- **Threshold sanity check**: If no ambiguity regime appears, τ_low and τ_high may be too close together. If no agreement appears, both agents may have very different biases.
