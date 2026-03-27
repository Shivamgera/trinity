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
