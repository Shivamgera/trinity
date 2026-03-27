"""Guardian Agent — composite of Stage 1 (hard constraints) and Stage 2 (adaptive policy).

Pipeline:
  proposed_action -> Stage 1 (hard constraints) -> if rejected, return override immediately
                                                 -> if allowed, Stage 2 (Delta-aware) -> final action
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """Composite Guardian agent combining both stages.

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
        cgate_result: Any = None,
        reasoning: str = "",
    ) -> FinalAction:
        """Run the full Guardian pipeline.

        Args:
            proposed_action: Action proposed by C-Gate (or directly by Executor).
            portfolio_state: Current portfolio state.
            cgate_result: Output from C-Gate (needs .regime attribute for Stage 2).
                          If None, Stage 2 is skipped (treated as agreement).
            reasoning: Analyst reasoning trace for audit logging during conflict.
                       Passed separately because CGateResult does not store it.

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
            stage2_result = self.stage2.apply(
                action=proposed_action,
                regime=regime,
                reasoning=reasoning,
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
