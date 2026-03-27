"""Guardian Stage 1: Hard constraints — unconditional risk limits.

These constraints fire BEFORE the C-Gate. They enforce portfolio-level
safety regardless of agent signals or regime classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


class GuardianConfig(BaseModel):
    """Configuration for hard constraint thresholds."""

    max_position_size: float = 1.0  # max abs position as fraction of portfolio value
    max_daily_loss: float = 0.05  # 5% max daily loss -> circuit breaker
    max_drawdown: float = 0.15  # 15% max drawdown -> circuit breaker
    min_cash_reserve: float = 0.1  # 10% minimum cash reserve


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
    """Stage 1 Guardian: enforces unconditional hard constraints.

    This guardian fires BEFORE the C-Gate — it does not know about
    Delta or the regime classification. Its sole purpose is to prevent
    catastrophic losses by enforcing portfolio-level risk limits.

    If ANY constraint is violated, the proposed action is rejected and
    overridden to flat (action=0), which is the safe default.
    """

    def __init__(self, config: GuardianConfig | None = None):
        self.config = config or GuardianConfig()

    def check(
        self, proposed_action: int, portfolio_state: PortfolioState
    ) -> ConstraintResult:
        """Check whether the proposed action satisfies all hard constraints.

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
                    f"max_drawdown: current drawdown "
                    f"({portfolio_state.current_drawdown:.2%}) "
                    f"exceeds limit ({self.config.max_drawdown:.2%})"
                )

        # 3. Position size limit
        #    Prevent entering a position that would exceed max position size.
        if proposed_action in (1, 2):  # long or short
            if abs(portfolio_state.position) >= self.config.max_position_size:
                violations.append(
                    f"max_position_size: current position "
                    f"({portfolio_state.position:.2f}) "
                    f"already at limit ({self.config.max_position_size:.2f})"
                )

        # 4. Cash reserve
        #    Don't enter long positions if cash reserve is below minimum.
        if proposed_action == 1:  # long (requires cash to buy)
            min_cash = self.config.min_cash_reserve * portfolio_state.portfolio_value
            if portfolio_state.cash < min_cash:
                violations.append(
                    f"min_cash_reserve: cash ({portfolio_state.cash:.2f}) "
                    f"below minimum ({min_cash:.2f})"
                )

        if violations:
            return ConstraintResult(
                allowed=False,
                violations=violations,
                override_action=0,  # flat = safe default
            )

        return ConstraintResult(allowed=True, violations=[], override_action=None)
