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


# Minimal mock of CGateResult — matches actual fields: delta, regime, action
@dataclass
class MockCGateResult:
    delta: float
    regime: str
    action: int


class TestAdaptiveGuardian:
    @pytest.fixture
    def adaptive(self):
        return AdaptiveGuardian(AdaptiveConfig())

    def test_agreement_passthrough(self, adaptive):
        result = adaptive.apply(1, "agreement")
        assert result.final_action == 1
        assert result.position_scale == 1.0
        assert result.stop_loss is None
        assert result.reasoning_logged is False

    def test_ambiguity_reduces_position(self, adaptive):
        result = adaptive.apply(1, "ambiguity")
        assert result.final_action == 1  # action preserved
        assert result.position_scale == 0.5  # reduced
        assert result.stop_loss == 0.02
        assert result.log_entry is not None

    def test_conflict_forces_flat(self, adaptive):
        result = adaptive.apply(1, "conflict")
        assert result.final_action == 0  # overridden to flat
        assert result.position_scale == 0.0

    def test_conflict_logs_reasoning(self, adaptive):
        result = adaptive.apply(
            2, "conflict", reasoning="Bearish signal from earnings miss"
        )
        assert result.reasoning_logged is True
        assert "earnings miss" in result.log_entry

    def test_conflict_no_reasoning(self, adaptive):
        result = adaptive.apply(1, "conflict", reasoning="")
        assert result.final_action == 0
        assert result.reasoning_logged is False

    def test_unknown_regime_raises(self, adaptive):
        with pytest.raises(ValueError, match="Unknown regime"):
            adaptive.apply(1, "unknown_regime")


class TestGuardianPipeline:
    """Test the full Guardian pipeline (Stage 1 + Stage 2)."""

    @pytest.fixture
    def guardian(self):
        return Guardian()

    def test_normal_trade_agreement(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.05, regime="agreement", action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 1
        assert result.position_scale == 1.0
        assert result.blocked_by_stage1 is False

    def test_normal_trade_ambiguity(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.25, regime="ambiguity", action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 1  # action preserved
        assert result.position_scale == 0.5  # reduced
        assert result.blocked_by_stage1 is False

    def test_normal_trade_conflict(self, guardian):
        state = make_state()
        cgate = MockCGateResult(delta=0.6, regime="conflict", action=0)
        result = guardian.process(0, state, cgate)
        assert result.action == 0  # flat
        assert result.blocked_by_stage1 is False

    def test_stage1_overrides_stage2(self, guardian):
        """Stage 1 rejection takes precedence over Stage 2."""
        state = make_state(daily_pnl=-10000)  # circuit breaker active
        cgate = MockCGateResult(delta=0.05, regime="agreement", action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 0  # forced flat by Stage 1
        assert result.blocked_by_stage1 is True
        assert result.stage2_result is None  # Stage 2 never ran

    def test_stage1_blocks_during_agreement(self, guardian):
        """Even in agreement regime, Stage 1 hard constraints still apply."""
        state = make_state(current_drawdown=0.20)  # drawdown exceeded
        cgate = MockCGateResult(delta=0.02, regime="agreement", action=1)
        result = guardian.process(1, state, cgate)
        assert result.action == 0
        assert result.blocked_by_stage1 is True

    def test_flat_always_allowed_through_pipeline(self, guardian):
        """Going flat passes both stages regardless of state."""
        state = make_state(daily_pnl=-20000, current_drawdown=0.30)
        cgate = MockCGateResult(delta=0.8, regime="conflict", action=0)
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
            cgate = MockCGateResult(delta=0.2, regime=regime, action=1)
            result = guardian.process(1, state, cgate)
            assert result.position_scale == expected_scale

        cgate = MockCGateResult(delta=0.6, regime="conflict", action=0)
        result = guardian.process(0, state, cgate)
        assert result.action == 0
        assert result.position_scale == 0.0

    def test_reasoning_forwarded_to_stage2(self, guardian):
        """Reasoning string should be passed through to Stage 2 for logging."""
        state = make_state()
        cgate = MockCGateResult(delta=0.6, regime="conflict", action=0)
        result = guardian.process(
            1, state, cgate, reasoning="Bearish earnings outlook"
        )
        assert result.stage2_result is not None
        assert result.stage2_result.reasoning_logged is True
        assert "Bearish earnings outlook" in result.stage2_result.log_entry
