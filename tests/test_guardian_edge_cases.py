"""Edge case tests for Guardian — boundary conditions, regime transitions, config loading."""

import pytest
from dataclasses import dataclass

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


@dataclass
class MockCGateResult:
    delta: float
    regime: str
    action: int


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

    def test_cash_exactly_at_reserve(self, guardian):
        """Cash exactly at 10% reserve should pass (strict <)."""
        state = make_state(cash=10000.0)  # exactly 10% of 100k
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is False

    def test_cash_one_cent_below_reserve(self, guardian):
        """Cash one cent below 10% should trigger."""
        state = make_state(cash=9999.99)
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True


class TestRegimeTransitions:
    """Test Guardian behavior across rapid regime transitions."""

    @pytest.fixture
    def guardian(self):
        return Guardian()

    def test_agreement_to_conflict_to_agreement(self, guardian):
        """Simulate rapid regime transitions over 3 timesteps."""
        state = make_state()

        # Step 1: Agreement — full pass-through
        r1 = guardian.process(
            1, state, MockCGateResult(0.05, "agreement", 1)
        )
        assert r1.action == 1
        assert r1.position_scale == 1.0

        # Step 2: Conflict — forced flat
        r2 = guardian.process(
            1, state, MockCGateResult(0.6, "conflict", 0)
        )
        assert r2.action == 0
        assert r2.position_scale == 0.0

        # Step 3: Back to agreement — should fully recover
        r3 = guardian.process(
            1, state, MockCGateResult(0.03, "agreement", 1)
        )
        assert r3.action == 1
        assert r3.position_scale == 1.0

    def test_ambiguity_to_agreement(self, guardian):
        """Transition from ambiguity to agreement restores full sizing."""
        state = make_state()

        r1 = guardian.process(
            1, state, MockCGateResult(0.25, "ambiguity", 1)
        )
        assert r1.position_scale == 0.5

        r2 = guardian.process(
            1, state, MockCGateResult(0.05, "agreement", 1)
        )
        assert r2.position_scale == 1.0

    def test_all_regimes_in_sequence(self, guardian):
        """Run through all three regimes in sequence."""
        state = make_state()
        regimes_and_scales = [
            ("agreement", 1.0),
            ("ambiguity", 0.5),
            ("conflict", 0.0),
            ("ambiguity", 0.5),
            ("agreement", 1.0),
        ]
        for regime, expected_scale in regimes_and_scales:
            cgate = MockCGateResult(0.2, regime, 1)
            result = guardian.process(1, state, cgate)
            assert result.position_scale == expected_scale, (
                f"Expected scale {expected_scale} for regime {regime}, "
                f"got {result.position_scale}"
            )


class TestAllConstraintsViolated:
    """Test behavior when all constraints are violated simultaneously."""

    def test_all_four_violations(self):
        guardian = Guardian()
        state = make_state(
            daily_pnl=-10000.0,  # daily loss exceeded
            current_drawdown=0.25,  # drawdown exceeded
            position=1.0,  # position at max
            cash=5000.0,  # cash below reserve
        )
        result = guardian.process(1, state)
        assert result.blocked_by_stage1 is True
        assert len(result.stage1_result.violations) >= 3  # at least 3 of 4
        assert result.action == 0

    def test_all_violations_with_conflict_regime(self):
        """Even if C-Gate says conflict, Stage 1 blocks first."""
        guardian = Guardian()
        state = make_state(
            daily_pnl=-10000.0,
            current_drawdown=0.25,
        )
        cgate = MockCGateResult(0.8, "conflict", 0)
        result = guardian.process(1, state, cgate)
        assert result.blocked_by_stage1 is True
        assert result.stage2_result is None  # Stage 2 never ran


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

    def test_load_project_config(self):
        """Load the actual configs/guardian.yaml and verify defaults."""
        hard, adaptive = load_guardian_config("configs/guardian.yaml")
        assert hard.max_position_size == 1.0
        assert hard.max_daily_loss == 0.05
        assert hard.max_drawdown == 0.15
        assert hard.min_cash_reserve == 0.10
        assert adaptive.ambiguity_position_scale == 0.5
        assert adaptive.ambiguity_stop_loss == 0.02
        assert adaptive.conflict_action == 0
