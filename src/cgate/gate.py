"""Consistency Gate — core arbitration logic for the Robust Trinity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.cgate.divergence import compute_delta


@dataclass
class CGateResult:
    """Output of the Consistency Gate evaluation."""

    delta: float  # Divergence Δ = 1 - π_RL(d_LLM)
    regime: Literal["agreement", "ambiguity", "conflict"]  # classified regime
    action: int  # final action: 0=flat, 1=long, 2=short


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
                f"Thresholds must satisfy 0 <= tau_low <= tau_high <= 1, "
                f"got tau_low={tau_low}, tau_high={tau_high}"
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
