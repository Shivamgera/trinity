# Phase 5: Guardian Agent — Rule Engine

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Project Root:** /Users/shivamgera/projects/research1
**Timeline:** Weeks 7–8 (~8–10 hours total across 3 tasks)

---

## P5-T1: Implement Guardian Stage 1 — Hard Constraints

**Estimated time:** ~2.5 hours
**Dependencies:** None (self-contained rule engine; only uses standard Python + Pydantic)

### Context

The Guardian is the third agent in the Robust Trinity. Unlike the Analyst (LLM-based) and Executor (RL-based), the Guardian is entirely rule-based. It serves as the system's safety net, enforcing risk management constraints that neither the Analyst nor the Executor can override.

The Guardian has **two stages** that fire at different points in the pipeline:

1. **Stage 1 — Hard Constraints (pre-C-Gate)**: Unconditional risk limits that apply regardless of what the Analyst and Executor say. These are portfolio-level safety checks (position limits, loss limits, drawdown circuit breakers). Stage 1 fires BEFORE the C-Gate even computes Δ.

2. **Stage 2 — Δ-Aware Adaptive Policy (post-C-Gate)**: Adjusts position sizing and stop-losses based on the regime (agreement/ambiguity/conflict) from the C-Gate. Built in P5-T2.

This task (P5-T1) implements Stage 1 only.

**Key design facts:**
- Action space: {flat=0, long=1, short=2} — same as Executor and C-Gate
- Stage 1 is unconditional — it does NOT know about Δ or the C-Gate regime
- If Stage 1 rejects an action, the pipeline short-circuits — Stage 2 and C-Gate output are irrelevant
- Hard constraints are conservative defaults; the human can tune them later

**Already available:**
- The project structure and environment from Phase 0
- Pydantic v2 is installed

### Objective

Implement the hard constraint guardian with configurable risk limits and portfolio state tracking.

### Detailed Instructions

#### Step 1: Create `src/guardian/__init__.py`

Leave empty for now — the composite Guardian class will be added in P5-T2.

#### Step 2: Create `src/guardian/hard_constraints.py`

```python
"""Guardian Stage 1: Hard constraints — unconditional risk limits.

These constraints fire BEFORE the C-Gate. They enforce portfolio-level
safety regardless of agent signals or regime classification.
"""
from dataclasses import dataclass, field
from pydantic import BaseModel


class GuardianConfig(BaseModel):
    """Configuration for hard constraint thresholds."""
    max_position_size: float = 1.0     # max abs position as fraction of portfolio value
    max_daily_loss: float = 0.05       # 5% max daily loss → circuit breaker
    max_drawdown: float = 0.15         # 15% max drawdown → circuit breaker
    min_cash_reserve: float = 0.1      # 10% minimum cash reserve


@dataclass
class PortfolioState:
    """Current state of the portfolio, used for constraint checking.

    Attributes:
        position: Current position as a fraction of portfolio value.
                  0.0 = flat, >0 = long, <0 = short.
        cash: Current cash as a fraction of portfolio value.
        portfolio_value: Current total portfolio value (cash + holdings).
        daily_pnl: Today's profit/loss as absolute value.
        peak_value: Historical peak portfolio value (for drawdown calculation).
        current_drawdown: (peak_value - portfolio_value) / peak_value.
                          0.0 = at peak, 0.15 = 15% below peak.
    """
    position: float
    cash: float
    portfolio_value: float
    daily_pnl: float
    peak_value: float
    current_drawdown: float


@dataclass
class ConstraintResult:
    """Result of hard constraint checking.

    Attributes:
        allowed: Whether the proposed action is permitted.
        violations: List of constraint names that were violated.
        override_action: If not allowed, the action to take instead (usually 0 = flat).
                         None if the action is allowed.
    """
    allowed: bool
    violations: list[str] = field(default_factory=list)
    override_action: int | None = None


# Action labels for readable violation messages
ACTION_LABELS = {0: "flat", 1: "long", 2: "short"}


class HardConstraintGuardian:
    """
    Stage 1 Guardian: enforces unconditional hard constraints.

    This guardian fires BEFORE the C-Gate — it does not know about Δ
    or the regime classification. Its sole purpose is to prevent
    catastrophic losses by enforcing portfolio-level risk limits.

    If ANY constraint is violated, the proposed action is rejected and
    overridden to flat (action=0), which is the safe default.
    """

    def __init__(self, config: GuardianConfig | None = None):
        self.config = config or GuardianConfig()

    def check(self, proposed_action: int, portfolio_state: PortfolioState) -> ConstraintResult:
        """
        Check whether the proposed action satisfies all hard constraints.

        Args:
            proposed_action: Proposed action from C-Gate or Executor.
                             0=flat, 1=long, 2=short.
            portfolio_state: Current portfolio state.

        Returns:
            ConstraintResult indicating whether the action is allowed.
        """
        violations = []

        # 1. Daily loss circuit breaker
        #    If daily losses exceed threshold, halt ALL trading.
        max_loss_abs = self.config.max_daily_loss * portfolio_state.portfolio_value
        if portfolio_state.daily_pnl < -max_loss_abs:
            if proposed_action != 0:  # only violates if trying to trade
                violations.append(
                    f"daily_loss_limit: daily PnL ({portfolio_state.daily_pnl:.2f}) "
                    f"exceeds max loss ({-max_loss_abs:.2f})"
                )

        # 2. Drawdown circuit breaker
        #    If drawdown exceeds threshold, halt ALL trading.
        if portfolio_state.current_drawdown > self.config.max_drawdown:
            if proposed_action != 0:
                violations.append(
                    f"max_drawdown: current drawdown ({portfolio_state.current_drawdown:.2%}) "
                    f"exceeds limit ({self.config.max_drawdown:.2%})"
                )

        # 3. Position size limit
        #    Prevent entering a position that would exceed max position size.
        if proposed_action in (1, 2):  # long or short
            if abs(portfolio_state.position) >= self.config.max_position_size:
                violations.append(
                    f"max_position_size: current position ({portfolio_state.position:.2f}) "
                    f"already at limit ({self.config.max_position_size:.2f})"
                )

        # 4. Cash reserve
        #    Don't enter long positions if cash reserve is below minimum.
        if proposed_action == 1:  # long (requires cash to buy)
            if portfolio_state.cash < self.config.min_cash_reserve * portfolio_state.portfolio_value:
                violations.append(
                    f"min_cash_reserve: cash ({portfolio_state.cash:.2f}) "
                    f"below minimum ({self.config.min_cash_reserve * portfolio_state.portfolio_value:.2f})"
                )

        if violations:
            return ConstraintResult(
                allowed=False,
                violations=violations,
                override_action=0,  # flat = safe default
            )

        return ConstraintResult(allowed=True, violations=[], override_action=None)
```

#### Step 3: Create tests in `tests/test_guardian.py`

```python
"""Tests for Guardian Stage 1 — Hard Constraints."""
import pytest
from src.guardian.hard_constraints import (
    HardConstraintGuardian,
    GuardianConfig,
    PortfolioState,
)


def make_state(**kwargs) -> PortfolioState:
    """Helper to create a PortfolioState with sensible defaults."""
    defaults = {
        "position": 0.0,
        "cash": 50000.0,
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "peak_value": 100000.0,
        "current_drawdown": 0.0,
    }
    defaults.update(kwargs)
    return PortfolioState(**defaults)


class TestHardConstraintGuardian:
    @pytest.fixture
    def guardian(self):
        return HardConstraintGuardian(GuardianConfig())

    # ---- Tests for valid trades passing through ----

    def test_valid_long_passes(self, guardian):
        state = make_state()
        result = guardian.check(1, state)
        assert result.allowed is True
        assert result.violations == []
        assert result.override_action is None

    def test_valid_short_passes(self, guardian):
        state = make_state()
        result = guardian.check(2, state)
        assert result.allowed is True

    def test_flat_always_passes(self, guardian):
        """Going flat should always be allowed, even during circuit breakers."""
        state = make_state(daily_pnl=-10000, current_drawdown=0.20)
        result = guardian.check(0, state)
        assert result.allowed is True

    # ---- Tests for daily loss circuit breaker ----

    def test_daily_loss_blocks_long(self, guardian):
        state = make_state(daily_pnl=-6000)  # > 5% of 100k
        result = guardian.check(1, state)
        assert result.allowed is False
        assert result.override_action == 0
        assert any("daily_loss_limit" in v for v in result.violations)

    def test_daily_loss_blocks_short(self, guardian):
        state = make_state(daily_pnl=-6000)
        result = guardian.check(2, state)
        assert result.allowed is False

    def test_daily_loss_at_threshold_passes(self, guardian):
        """PnL exactly at -5% should pass (we check < -max_loss_abs)."""
        state = make_state(daily_pnl=-5000)  # exactly 5%
        result = guardian.check(1, state)
        assert result.allowed is True

    # ---- Tests for drawdown circuit breaker ----

    def test_drawdown_blocks_trading(self, guardian):
        state = make_state(
            current_drawdown=0.20,  # 20% > 15% limit
            peak_value=120000,
            portfolio_value=96000,
        )
        result = guardian.check(1, state)
        assert result.allowed is False
        assert any("max_drawdown" in v for v in result.violations)

    def test_drawdown_at_limit_passes(self, guardian):
        state = make_state(current_drawdown=0.15)  # exactly at limit
        result = guardian.check(1, state)
        assert result.allowed is True

    # ---- Tests for position size ----

    def test_position_limit_blocks(self, guardian):
        state = make_state(position=1.0)  # already at max
        result = guardian.check(1, state)
        assert result.allowed is False
        assert any("max_position_size" in v for v in result.violations)

    def test_position_limit_allows_flat(self, guardian):
        """Even if at max position, going flat should be allowed."""
        state = make_state(position=1.0)
        result = guardian.check(0, state)
        assert result.allowed is True

    # ---- Tests for cash reserve ----

    def test_low_cash_blocks_long(self, guardian):
        state = make_state(cash=5000)  # 5% < 10% reserve
        result = guardian.check(1, state)
        assert result.allowed is False
        assert any("min_cash_reserve" in v for v in result.violations)

    def test_low_cash_allows_short(self, guardian):
        """Short doesn't require buying, so low cash shouldn't block it."""
        state = make_state(cash=5000)
        result = guardian.check(2, state)
        # Cash reserve only blocks long (buying), not short
        cash_violations = [v for v in result.violations if "min_cash_reserve" in v]
        assert len(cash_violations) == 0

    # ---- Tests for multiple violations ----

    def test_multiple_violations(self, guardian):
        state = make_state(
            daily_pnl=-10000,   # daily loss violated
            current_drawdown=0.20,  # drawdown violated
            cash=5000,          # cash violated
        )
        result = guardian.check(1, state)
        assert result.allowed is False
        assert len(result.violations) >= 2

    # ---- Custom config tests ----

    def test_custom_config(self):
        config = GuardianConfig(
            max_daily_loss=0.01,  # 1% — very tight
            max_drawdown=0.05,
            min_cash_reserve=0.2,
        )
        guardian = HardConstraintGuardian(config)
        state = make_state(daily_pnl=-1500)  # 1.5% > 1% limit
        result = guardian.check(1, state)
        assert result.allowed is False
```

### Acceptance Criteria

- [ ] `HardConstraintGuardian` correctly enforces all four constraints: daily loss, drawdown, position size, cash reserve
- [ ] Going flat (action=0) is ALWAYS allowed, even when circuit breakers are active
- [ ] Multiple simultaneous violations are all reported
- [ ] Override action is always 0 (flat) when constraints are violated
- [ ] `GuardianConfig` is a Pydantic model with sensible defaults
- [ ] All tests pass: `pytest tests/test_guardian.py -v`

### Files to Create/Modify

- `src/guardian/__init__.py` (create, empty for now)
- `src/guardian/hard_constraints.py` (create)
- `tests/test_guardian.py` (create)

### Dependencies

- None — this is a self-contained rule engine

### Human Checkpoint

- Review the constraint thresholds in `GuardianConfig`: are 5% daily loss, 15% drawdown, and 10% cash reserve reasonable for the trading strategy? Adjust if the backtest data suggests different risk parameters.

---

## P5-T2: Implement Guardian Stage 2 — Δ-Aware Adaptive Policy

**Estimated time:** ~2.5 hours
**Dependencies:** P5-T1 must be complete (Stage 1 classes exist); P4-T1 must be complete (CGateResult dataclass exists)

### Context

Stage 1 of the Guardian (built in P5-T1) enforces unconditional hard constraints that fire before the C-Gate. Stage 2 fires AFTER the C-Gate and adapts its behavior based on the regime classification (agreement, ambiguity, conflict).

The key insight: when the Analyst and Executor disagree (high Δ), the system should become more conservative even if no hard constraint is violated. Stage 2 implements this by:
- **Agreement**: Pass through unchanged. Full position sizing.
- **Ambiguity**: Reduce position size (e.g., to 50%), apply tighter stop-losses. The trade is still allowed but with reduced risk exposure.
- **Conflict**: Override to flat. Log the Analyst's reasoning trace for audit. The reasoning trace `r` is for human inspection only — it does NOT affect the decision programmatically.

After Stage 2, the Guardian composes both stages into a single pipeline: Stage 1 check → Stage 2 adaptive policy → final action.

**Already available from P5-T1:**
- `src/guardian/hard_constraints.py` — `HardConstraintGuardian`, `GuardianConfig`, `PortfolioState`, `ConstraintResult`

**Already available from P4-T1:**
- `src/cgate/gate.py` — `CGateResult` dataclass with fields: `delta`, `regime`, `merged_action`, `merged_distribution`, `reasoning_forwarded`

### Objective

Implement the Δ-aware adaptive guardian and compose it with Stage 1 into the final Guardian pipeline.

### Detailed Instructions

#### Step 1: Create `src/guardian/adaptive_policy.py`

```python
"""Guardian Stage 2: Δ-aware adaptive policy — regime-dependent risk adjustment.

This stage fires AFTER the C-Gate. It adjusts position sizing and stop-losses
based on the regime (agreement/ambiguity/conflict).
"""
from dataclasses import dataclass
from pydantic import BaseModel


class AdaptiveConfig(BaseModel):
    """Configuration for Δ-aware adaptive behavior."""
    ambiguity_position_scale: float = 0.5   # scale position to 50% during ambiguity
    ambiguity_stop_loss: float = 0.02       # 2% tighter stop-loss during ambiguity
    conflict_action: int = 0                # flat — fail-safe during conflict


@dataclass
class AdaptiveResult:
    """Result of adaptive policy application.

    Attributes:
        final_action: The action after adaptive adjustments.
                      0=flat, 1=long, 2=short.
        position_scale: Scaling factor for position size (1.0 = full, 0.5 = half, 0.0 = none).
        stop_loss: Stop-loss threshold override, or None if no override.
        reasoning_logged: Whether the Analyst's reasoning was logged for audit.
        log_entry: Human-readable log message, or None.
    """
    final_action: int
    position_scale: float
    stop_loss: float | None
    reasoning_logged: bool
    log_entry: str | None


class AdaptiveGuardian:
    """
    Stage 2 Guardian: Δ-aware adaptive risk policy.

    Adjusts position sizing and risk parameters based on the C-Gate
    regime classification. The Analyst's reasoning trace is only
    logged during conflict — it does NOT affect the decision.
    """

    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()

    def apply(
        self,
        action: int,
        regime: str,
        portfolio_state: "PortfolioState",  # forward ref to avoid circular import
        reasoning: str = "",
    ) -> AdaptiveResult:
        """
        Apply adaptive policy based on regime.

        Args:
            action: The merged action from the C-Gate (0=flat, 1=long, 2=short).
            regime: C-Gate regime: "agreement", "ambiguity", or "conflict".
            portfolio_state: Current portfolio state (for context in logging).
            reasoning: Analyst reasoning trace (logged only during conflict).

        Returns:
            AdaptiveResult with potentially modified action and risk parameters.
        """
        if regime == "agreement":
            return AdaptiveResult(
                final_action=action,
                position_scale=1.0,
                stop_loss=None,
                reasoning_logged=False,
                log_entry=None,
            )

        elif regime == "ambiguity":
            return AdaptiveResult(
                final_action=action,  # keep the action, but reduce position
                position_scale=self.config.ambiguity_position_scale,
                stop_loss=self.config.ambiguity_stop_loss,
                reasoning_logged=False,
                log_entry=(
                    f"[AMBIGUITY] Action {action} allowed at reduced scale "
                    f"({self.config.ambiguity_position_scale:.0%}), "
                    f"stop-loss tightened to {self.config.ambiguity_stop_loss:.1%}"
                ),
            )

        elif regime == "conflict":
            log_msg = (
                f"[CONFLICT] Action overridden to {self.config.conflict_action} (flat). "
                f"Original action: {action}."
            )
            if reasoning:
                log_msg += f"\n  Analyst reasoning: {reasoning[:500]}"

            return AdaptiveResult(
                final_action=self.config.conflict_action,
                position_scale=0.0,
                stop_loss=None,
                reasoning_logged=bool(reasoning),
                log_entry=log_msg,
            )

        else:
            raise ValueError(f"Unknown regime: {regime}. Expected agreement/ambiguity/conflict.")
```

#### Step 2: Update `src/guardian/__init__.py` with composite Guardian

```python
"""Guardian Agent — composite of Stage 1 (hard constraints) and Stage 2 (adaptive policy).

Pipeline:
  proposed_action → Stage 1 (hard constraints) → if rejected, return override immediately
                                                → if allowed, Stage 2 (Δ-aware) → final action
"""
from dataclasses import dataclass

from src.guardian.hard_constraints import (
    HardConstraintGuardian,
    GuardianConfig,
    PortfolioState,
    ConstraintResult,
)
from src.guardian.adaptive_policy import (
    AdaptiveGuardian,
    AdaptiveConfig,
    AdaptiveResult,
)


@dataclass
class FinalAction:
    """Complete Guardian output after both stages.

    Attributes:
        action: Final action to execute (0=flat, 1=long, 2=short).
        position_scale: Position sizing factor (1.0 = full, 0.5 = half, 0.0 = none).
        stop_loss: Stop-loss override, or None.
        stage1_result: Result from Stage 1 (hard constraints).
        stage2_result: Result from Stage 2 (adaptive policy), or None if Stage 1 rejected.
        blocked_by_stage1: Whether Stage 1 rejected the action.
    """
    action: int
    position_scale: float
    stop_loss: float | None
    stage1_result: ConstraintResult
    stage2_result: AdaptiveResult | None
    blocked_by_stage1: bool


class Guardian:
    """
    Composite Guardian agent combining both stages.

    Stage 1 (hard constraints) fires unconditionally.
    Stage 2 (adaptive policy) only fires if Stage 1 allows the action.

    If Stage 1 rejects, the final action is the Stage 1 override (flat),
    regardless of what Stage 2 would have done.
    """

    def __init__(
        self,
        hard_config: GuardianConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        self.stage1 = HardConstraintGuardian(hard_config)
        self.stage2 = AdaptiveGuardian(adaptive_config)

    def process(
        self,
        proposed_action: int,
        portfolio_state: PortfolioState,
        cgate_result: "CGateResult | None" = None,
    ) -> FinalAction:
        """
        Run the full Guardian pipeline.

        Args:
            proposed_action: Action proposed by C-Gate (or directly by Executor).
            portfolio_state: Current portfolio state.
            cgate_result: Output from C-Gate (needed for Stage 2).
                          If None, Stage 2 is skipped (treated as agreement).

        Returns:
            FinalAction with the ultimate action to execute.
        """
        # Stage 1: Hard constraints (unconditional)
        stage1_result = self.stage1.check(proposed_action, portfolio_state)

        if not stage1_result.allowed:
            # Stage 1 rejected — short-circuit, don't even run Stage 2
            return FinalAction(
                action=stage1_result.override_action,
                position_scale=0.0,
                stop_loss=None,
                stage1_result=stage1_result,
                stage2_result=None,
                blocked_by_stage1=True,
            )

        # Stage 2: Adaptive policy (regime-dependent)
        if cgate_result is not None:
            regime = cgate_result.regime
            reasoning = "" if not cgate_result.reasoning_forwarded else ""
            # Note: reasoning text is not stored in CGateResult — it's the caller's
            # responsibility to pass it. In the full pipeline, the orchestrator
            # will pass the reasoning from the Analyst signal.
            stage2_result = self.stage2.apply(
                action=proposed_action,
                regime=regime,
                portfolio_state=portfolio_state,
                reasoning="",  # Will be wired up by orchestrator in later phases
            )
        else:
            # No C-Gate result — treat as agreement (pass through)
            stage2_result = AdaptiveResult(
                final_action=proposed_action,
                position_scale=1.0,
                stop_loss=None,
                reasoning_logged=False,
                log_entry=None,
            )

        return FinalAction(
            action=stage2_result.final_action,
            position_scale=stage2_result.position_scale,
            stop_loss=stage2_result.stop_loss,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            blocked_by_stage1=False,
        )
```

#### Step 3: Extend tests with Stage 2 and pipeline tests

Create `tests/test_guardian_pipeline.py`:

```python
"""Tests for Guardian Stage 2 (adaptive policy) and full pipeline."""
import pytest
from dataclasses import dataclass
from src.guardian import Guardian, FinalAction
from src.guardian.hard_constraints import (
    GuardianConfig,
    PortfolioState,
)
from src.guardian.adaptive_policy import AdaptiveGuardian, AdaptiveConfig


def make_state(**kwargs) -> PortfolioState:
    defaults = {
        "position": 0.0,
        "cash": 50000.0,
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "peak_value": 100000.0,
        "current_drawdown": 0.0,
    }
    defaults.update(kwargs)
    return PortfolioState(**defaults)


# Minimal mock of CGateResult to avoid importing from cgate
@dataclass
class MockCGateResult:
    delta: float
    regime: str
    merged_action: int
    merged_distribution: object = None
    reasoning_forwarded: bool = False


class TestAdaptiveGuardian:
    @pytest.fixture
    def adaptive(self):
        return AdaptiveGuardian(AdaptiveConfig())

    def test_agreement_passthrough(self, adaptive):
        state = make_state()
        result = adaptive.apply(1, "agreement", state)
        assert result.final_action == 1
        assert result.position_scale == 1.0
        assert result.stop_loss is None
        assert result.reasoning_logged is False

    def test_ambiguity_reduces_position(self, adaptive):
        state = make_state()
        result = adaptive.apply(1, "ambiguity", state)
        assert result.final_action == 1  # action preserved
        assert result.position_scale == 0.5  # reduced
        assert result.stop_loss == 0.02
        assert result.log_entry is not None

    def test_conflict_forces_flat(self, adaptive):
        state = make_state()
        result = adaptive.apply(1, "conflict", state)
        assert result.final_action == 0  # overridden to flat
        assert result.position_scale == 0.0

    def test_conflict_logs_reasoning(self, adaptive):
        state = make_state()
        result = adaptive.apply(2, "conflict", state, reasoning="Bearish signal from earnings miss")
        assert result.reasoning_logged is True
        assert "earnings miss" in result.log_entry

    def test_conflict_no_reasoning(self, adaptive):
        state = make_state()
        result = adaptive.apply(1, "conflict", state, reasoning="")
        assert result.final_action == 0
        assert result.reasoning_logged is False

    def test_unknown_regime_raises(self, adaptive):
        state = make_state()
        with pytest.raises(ValueError, match="Unknown regime"):
            adaptive.apply(1, "unknown_regime", state)


class TestGuardianPipeline:
    """Test the full Guardian pipeline (Stage 1 + Stage 2)."""

    @pytest.fixture
    def guardian(self):
        return Guardian()

    def test_normal_trade_agreement(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.05, regime="agreement", merged_action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 1
        assert result.position_scale == 1.0
        assert result.blocked_by_stage1 is False

    def test_normal_trade_ambiguity(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.25, regime="ambiguity", merged_action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 1  # action preserved
        assert result.position_scale == 0.5  # reduced
        assert result.blocked_by_stage1 is False

    def test_normal_trade_conflict(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.6, regime="conflict", merged_action=0)
        result = guardian.process(0, state, cgate)
        assert result.action == 0  # flat
        assert result.blocked_by_stage1 is False

    def test_stage1_overrides_stage2(self, guardian):
        """Stage 1 rejection takes precedence over Stage 2."""
        state = make_state(daily_pnl=-10000)  # circuit breaker active
        cgate = MockCGateResult(delta=0.05, regime="agreement", merged_action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 0  # forced flat by Stage 1
        assert result.blocked_by_stage1 is True
        assert result.stage2_result is None  # Stage 2 never ran

    def test_stage1_blocks_during_agreement(self, guardian):
        """Even in agreement regime, Stage 1 hard constraints still apply."""
        state = make_state(current_drawdown=0.20)  # drawdown exceeded
        cgate = MockCGateResult(delta=0.02, regime="agreement", merged_action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 0
        assert result.blocked_by_stage1 is True

    def test_flat_always_allowed_through_pipeline(self, guardian):
        """Going flat passes both stages regardless of state."""
        state = make_state(daily_pnl=-20000, current_drawdown=0.30)
        cgate = MockCGateResult(delta=0.8, regime="conflict", merged_action=0)
        result = guardian.process(0, state, cgate)
        assert result.action == 0
        assert result.blocked_by_stage1 is False  # flat is never blocked

    def test_no_cgate_result(self, guardian):
        """Without C-Gate result, Stage 2 defaults to pass-through."""
        state = make_state()
        result = guardian.process(1, state, cgate_result=None)
        assert result.action == 1
        assert result.position_scale == 1.0

    def test_all_three_regimes(self, guardian):
        """Test all three regimes produce valid outputs."""
        state = make_state()
        for regime, expected_scale in [("agreement", 1.0), ("ambiguity", 0.5)]:
            cgate = MockCGateResult(delta=0.2, regime=regime, merged_action=1)
            result = guardian.process(1, state, cgate)
            assert result.position_scale == expected_scale

        cgate = MockCGateResult(delta=0.6, regime="conflict", merged_action=0)
        result = guardian.process(0, state, cgate)
        assert result.action == 0
        assert result.position_scale == 0.0
```

### Acceptance Criteria

- [ ] `AdaptiveGuardian` correctly handles all three regimes
- [ ] Agreement: action passes through, full position scale
- [ ] Ambiguity: action preserved, position scale reduced, stop-loss applied
- [ ] Conflict: action overridden to flat, reasoning logged if present
- [ ] Composite `Guardian.process()` correctly sequences Stage 1 → Stage 2
- [ ] Stage 1 rejection short-circuits Stage 2 (Stage 2 never runs)
- [ ] All tests pass: `pytest tests/test_guardian.py tests/test_guardian_pipeline.py -v`

### Files to Create/Modify

- `src/guardian/adaptive_policy.py` (create)
- `src/guardian/__init__.py` (update — add composite Guardian class)
- `tests/test_guardian_pipeline.py` (create)

### Dependencies

- P5-T1 must be complete (`hard_constraints.py` exists)
- P4-T1 must be complete (`CGateResult` dataclass exists) — though we mock it in tests to avoid a hard import dependency

### Human Checkpoint

- Review the adaptive config defaults: is 50% position scaling during ambiguity reasonable? Is a 2% stop-loss tight enough? These should be adjusted based on the trading strategy's typical risk parameters.

---

## P5-T3: Guardian Configuration and Edge Case Testing

**Estimated time:** ~2 hours
**Dependencies:** P5-T1 and P5-T2 must be complete

### Context

The Guardian is now fully implemented with both stages. Before integrating it into the full orchestration pipeline, we need to:

1. **Externalize configuration** into a YAML file for easy tuning
2. **Test edge cases** that could occur in production (rapid regime transitions, simultaneous constraint violations, boundary conditions)
3. **Run an integration simulation** to verify the Guardian behaves correctly over a sequence of timesteps with varying conditions

**Already available:**
- `src/guardian/hard_constraints.py` — Stage 1 (hard constraints) from P5-T1
- `src/guardian/adaptive_policy.py` — Stage 2 (adaptive policy) from P5-T2
- `src/guardian/__init__.py` — Composite `Guardian` class from P5-T2
- `tests/test_guardian.py` — Stage 1 unit tests from P5-T1
- `tests/test_guardian_pipeline.py` — Pipeline tests from P5-T2

### Objective

Create a YAML configuration file, write edge case tests, and run a simulation to verify Guardian behavior under stress.

### Detailed Instructions

#### Step 1: Create `configs/guardian.yaml`

```yaml
# Guardian Agent Configuration
# =================================
# Stage 1: Hard Constraints (unconditional, pre-C-Gate)
# Stage 2: Δ-Aware Adaptive Policy (post-C-Gate)

hard_constraints:
  max_position_size: 1.0      # Max position as fraction of portfolio (1.0 = 100%)
  max_daily_loss: 0.05        # 5% daily loss → circuit breaker
  max_drawdown: 0.15          # 15% max drawdown → circuit breaker
  min_cash_reserve: 0.10      # 10% minimum cash reserve

adaptive_policy:
  ambiguity_position_scale: 0.5   # Reduce position to 50% during ambiguity
  ambiguity_stop_loss: 0.02       # 2% stop-loss during ambiguity
  conflict_action: 0               # Force flat (0) during conflict
```

#### Step 2: Create `src/guardian/config.py` — YAML loader

```python
"""Guardian configuration loader."""
import yaml
from pathlib import Path

from src.guardian.hard_constraints import GuardianConfig
from src.guardian.adaptive_policy import AdaptiveConfig


def load_guardian_config(
    config_path: str = "configs/guardian.yaml",
) -> tuple[GuardianConfig, AdaptiveConfig]:
    """
    Load Guardian configuration from a YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Tuple of (GuardianConfig, AdaptiveConfig).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Guardian config not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    hard_config = GuardianConfig(**raw.get("hard_constraints", {}))
    adaptive_config = AdaptiveConfig(**raw.get("adaptive_policy", {}))

    return hard_config, adaptive_config
```

#### Step 3: Create `tests/test_guardian_edge_cases.py`

Write edge case tests covering:

- **Boundary conditions**: values exactly at constraint thresholds (daily loss at exactly 5%, drawdown at exactly 15%)
- **Rapid regime transitions**: sequence of agreement → conflict → agreement within consecutive timesteps
- **Simultaneous violations**: all four hard constraints violated at once
- **Config validation**: invalid config values (negative thresholds, position > 1.0)
- **YAML round-trip**: load config from YAML, verify values match expected defaults

```python
"""Edge case tests for Guardian — boundary conditions, regime transitions, config loading."""
import pytest
import tempfile
from pathlib import Path

from src.guardian import Guardian, FinalAction
from src.guardian.hard_constraints import GuardianConfig, PortfolioState
from src.guardian.adaptive_policy import AdaptiveConfig
from src.guardian.config import load_guardian_config


def make_state(**kwargs) -> PortfolioState:
    defaults = {
        "position": 0.0,
        "cash": 50000.0,
        "portfolio_value": 100000.0,
        "daily_pnl": 0.0,
        "peak_value": 100000.0,
        "current_drawdown": 0.0,
    }
    defaults.update(kwargs)
    return PortfolioState(**defaults)


class TestBoundaryConditions:
    """Test exact boundary values for all constraints."""

    @pytest.fixture
    def guardian(self):
        return Guardian()

    def test_daily_loss_exactly_at_limit(self, guardian):
        """PnL exactly at -5% boundary (not exceeded) should pass."""
        state = make_state(daily_pnl=-5000.0)  # exactly 5% of 100k
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is False

    def test_daily_loss_one_cent_over(self, guardian):
        """PnL one cent past -5% should trigger circuit breaker."""
        state = make_state(daily_pnl=-5000.01)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True

    def test_drawdown_exactly_at_limit(self, guardian):
        """Drawdown exactly at 15% should pass (strict >)."""
        state = make_state(current_drawdown=0.15)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is False

    def test_drawdown_one_bp_over(self, guardian):
        """Drawdown one basis point over 15% should trigger."""
        state = make_state(current_drawdown=0.1501)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True

    def test_position_exactly_at_max(self, guardian):
        """Position exactly at 1.0 should block new trades."""
        state = make_state(position=1.0)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True

    def test_position_just_below_max(self, guardian):
        """Position just below 1.0 should allow trades."""
        state = make_state(position=0.99)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is False


class TestRegimeTransitions:
    """Test Guardian behavior across rapid regime transitions."""

    @pytest.fixture
    def guardian(self):
        return Guardian()

    def test_agreement_to_conflict_to_agreement(self, guardian):
        """Simulate rapid regime transitions over 3 timesteps."""
        from dataclasses import dataclass

        @dataclass
        class MockCGate:
            delta: float
            regime: str
            merged_action: int
            merged_distribution: object = None
            reasoning_forwarded: bool = False

        state = make_state()

        # Step 1: Agreement — full pass-through
        r1 = guardian.process(1, state, MockCGate(0.05, "agreement", 1))
        assert r1.action == 1
        assert r1.position_scale == 1.0

        # Step 2: Conflict — forced flat
        r2 = guardian.process(1, state, MockCGate(0.6, "conflict", 0))
        assert r2.action == 0
        assert r2.position_scale == 0.0

        # Step 3: Back to agreement — should fully recover
        r3 = guardian.process(1, state, MockCGate(0.03, "agreement", 1))
        assert r3.action == 1
        assert r3.position_scale == 1.0


class TestAllConstraintsViolated:
    """Test behavior when all constraints are violated simultaneously."""

    def test_all_four_violations(self):
        guardian = Guardian()
        state = make_state(
            daily_pnl=-10000.0,       # daily loss exceeded
            current_drawdown=0.25,    # drawdown exceeded
            position=1.0,             # position at max
            cash=5000.0,              # cash below reserve
        )
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True
        assert len(result.stage1_result.violations) >= 3  # at least 3 of 4
        assert result.action == 0


class TestConfigLoading:
    """Test YAML config loading and round-trip."""

    def test_load_default_config(self, tmp_path):
        config_file = tmp_path / "guardian.yaml"
        config_file.write_text(
            """
hard_constraints:
  max_position_size: 0.8
  max_daily_loss: 0.03
  max_drawdown: 0.10
  min_cash_reserve: 0.15

adaptive_policy:
  ambiguity_position_scale: 0.3
  ambiguity_stop_loss: 0.015
  conflict_action: 0
"""
        )
        hard, adaptive = load_guardian_config(str(config_file))
        assert hard.max_position_size == 0.8
        assert hard.max_daily_loss == 0.03
        assert hard.max_drawdown == 0.10
        assert hard.min_cash_reserve == 0.15
        assert adaptive.ambiguity_position_scale == 0.3
        assert adaptive.ambiguity_stop_loss == 0.015

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_guardian_config("/nonexistent/path/guardian.yaml")

    def test_partial_config_uses_defaults(self, tmp_path):
        config_file = tmp_path / "guardian.yaml"
        config_file.write_text(
            """
hard_constraints:
  max_daily_loss: 0.02

adaptive_policy: {}
"""
        )
        hard, adaptive = load_guardian_config(str(config_file))
        assert hard.max_daily_loss == 0.02
        assert hard.max_drawdown == 0.15  # default
        assert adaptive.ambiguity_position_scale == 0.5  # default
```

### Acceptance Criteria

- [ ] `configs/guardian.yaml` exists with documented defaults
- [ ] `load_guardian_config()` correctly loads and validates YAML config
- [ ] Boundary condition tests verify exact threshold behavior
- [ ] Regime transition tests confirm no state leakage between timesteps
- [ ] All-constraints-violated test confirms correct short-circuit behavior
- [ ] Config round-trip test confirms YAML values are correctly loaded
- [ ] All tests pass: `pytest tests/test_guardian.py tests/test_guardian_pipeline.py tests/test_guardian_edge_cases.py -v`

### Files to Create/Modify

- `configs/guardian.yaml` (create)
- `src/guardian/config.py` (create)
- `tests/test_guardian_edge_cases.py` (create)

### Dependencies

- P5-T1 and P5-T2 must be complete
- `pyyaml` must be installed (should already be in the environment)

### Human Checkpoint

- Review the YAML config values and confirm they match the risk parameters for your trading strategy
- Verify that the edge case tests cover the scenarios most relevant to your deployment environment
