"""Guardian Stage 2: Delta-aware adaptive policy — regime-dependent risk adjustment.

This stage fires AFTER the C-Gate. It adjusts position sizing and stop-losses
based on the regime (agreement/ambiguity/conflict).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


class AdaptiveConfig(BaseModel):
    """Configuration for Delta-aware adaptive behavior."""

    ambiguity_position_scale: float = 0.5  # scale position to 50% during ambiguity
    ambiguity_stop_loss: float = 0.02  # 2% tighter stop-loss during ambiguity
    conflict_action: int = 0  # flat -- fail-safe during conflict


@dataclass
class AdaptiveResult:
    """Result of adaptive policy application.

    Attributes:
        final_action: The action after adaptive adjustments.
                      0=flat, 1=long, 2=short.
        position_scale: Scaling factor for position size
                        (1.0 = full, 0.5 = half, 0.0 = none).
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
    """Stage 2 Guardian: Delta-aware adaptive risk policy.

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
        reasoning: str = "",
    ) -> AdaptiveResult:
        """Apply adaptive policy based on regime.

        Args:
            action: The merged action from the C-Gate (0=flat, 1=long, 2=short).
            regime: C-Gate regime: "agreement", "ambiguity", or "conflict".
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
                f"[CONFLICT] Action overridden to {self.config.conflict_action} "
                f"(flat). Original action: {action}."
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
            raise ValueError(
                f"Unknown regime: {regime}. "
                f"Expected agreement/ambiguity/conflict."
            )
