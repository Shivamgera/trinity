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
            daily_pnl=-10000,  # daily loss violated
            current_drawdown=0.20,  # drawdown violated
            cash=5000,  # cash violated
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
