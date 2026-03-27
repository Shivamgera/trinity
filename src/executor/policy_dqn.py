"""DQN policy extraction utilities for the Executor agent.

Mirrors src/executor/policy.py but for DQN's Q-network architecture.
DQN uses a single Q-network that outputs Q(s,a) for all actions,
whereas PPO has separate actor-critic heads via mlp_extractor.

For C-Gate integration, action probabilities are derived via
softmax(Q-values / temperature), which is functionally equivalent
to PPO's logit-based extraction.

The Q-Value Spread metric measures DQN decisiveness:
    spread = (max(Q) - mean(Q)) / std(Q)
A consistently high spread indicates the network has learned to
strongly prefer one action over others---analogous to low entropy
in PPO's policy distribution.

Usage:
    model, normalizer = load_executor_dqn("experiments/executor_dqn/frozen/seed_42")
    probs = get_policy_distribution(model, obs, normalizer)
    spread = compute_q_value_spread(model, obs, normalizer)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from src.executor.static_normalize import StaticNormalizer


def _extract_q_values(model: DQN, obs_tensor: torch.Tensor) -> torch.Tensor:
    """Extract Q-values for all actions from the DQN Q-network.

    Args:
        model: Trained DQN model.
        obs_tensor: Observation tensor of shape (B, obs_dim).

    Returns:
        Q-values tensor of shape (B, n_actions).
    """
    return model.q_net(obs_tensor)


def get_policy_distribution(
    model: DQN,
    obs: np.ndarray,
    normalizer: StaticNormalizer | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Get action probability distribution for a single observation.

    Probabilities are derived via softmax over Q-values, optionally
    scaled by temperature.  At T=1.0 (default), this gives the
    Boltzmann policy distribution.

    Args:
        model: Trained DQN model.
        obs: Raw observation of shape (obs_dim,) or (1, obs_dim).
        normalizer: Optional StaticNormalizer. If provided, obs is
                    normalized before being passed to the Q-network.
        temperature: Softmax temperature (default 1.0 = no scaling).
                     Values < 1.0 sharpen; > 1.0 flatten.  Must be > 0.

    Returns:
        ndarray of shape (3,) with probabilities [p_flat, p_long, p_short].
        Sums to 1.0.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if obs.ndim == 1:
        obs = obs[np.newaxis, :]

    obs = obs.astype(np.float32)

    if normalizer is not None:
        obs = normalizer.normalize_obs(obs)

    obs_tensor = torch.as_tensor(obs, device=model.policy.device)

    with torch.no_grad():
        q_values = _extract_q_values(model, obs_tensor)  # (1, 3)
        probs = torch.softmax(q_values / temperature, dim=-1)  # (1, 3)

    return probs.cpu().numpy().flatten()  # shape (3,)


def get_policy_distribution_batch(
    model: DQN,
    obs_batch: np.ndarray,
    normalizer: StaticNormalizer | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Get action probability distributions for a batch of observations.

    Args:
        model: Trained DQN model.
        obs_batch: Raw observations of shape (B, obs_dim).
        normalizer: Optional StaticNormalizer for normalization.
        temperature: Softmax temperature (default 1.0).  Must be > 0.

    Returns:
        ndarray of shape (B, 3) with probabilities.
        Each row sums to 1.0.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    obs_batch = obs_batch.astype(np.float32)

    if normalizer is not None:
        obs_batch = normalizer.normalize_obs(obs_batch)

    obs_tensor = torch.as_tensor(obs_batch, device=model.policy.device)

    with torch.no_grad():
        q_values = _extract_q_values(model, obs_tensor)  # (B, 3)
        probs = torch.softmax(q_values / temperature, dim=-1)  # (B, 3)

    return probs.cpu().numpy()  # shape (B, 3)


def compute_q_value_spread(
    model: DQN,
    obs: np.ndarray,
    normalizer: StaticNormalizer | None = None,
) -> float:
    """Compute Q-Value Spread for a single observation.

    Spread = (max(Q) - mean(Q)) / std(Q)

    A high spread means the network decisively prefers one action.
    A spread near zero means the Q-values are nearly uniform
    (indecisive / undertrained).

    Args:
        model: Trained DQN model.
        obs: Raw observation of shape (obs_dim,) or (1, obs_dim).
        normalizer: Optional StaticNormalizer.

    Returns:
        Q-Value Spread (scalar).  Returns 0.0 if std(Q) < 1e-12.
    """
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]

    obs = obs.astype(np.float32)

    if normalizer is not None:
        obs = normalizer.normalize_obs(obs)

    obs_tensor = torch.as_tensor(obs, device=model.policy.device)

    with torch.no_grad():
        q_values = _extract_q_values(model, obs_tensor)  # (1, 3)

    q_np = q_values.cpu().numpy().flatten()  # (3,)
    q_std = np.std(q_np)
    if q_std < 1e-12:
        return 0.0
    return float((np.max(q_np) - np.mean(q_np)) / q_std)


def compute_q_value_spread_batch(
    model: DQN,
    obs_batch: np.ndarray,
    normalizer: StaticNormalizer | None = None,
) -> np.ndarray:
    """Compute Q-Value Spread for a batch of observations.

    Args:
        model: Trained DQN model.
        obs_batch: Raw observations of shape (B, obs_dim).
        normalizer: Optional StaticNormalizer.

    Returns:
        ndarray of shape (B,) with per-observation Q-Value Spread.
    """
    obs_batch = obs_batch.astype(np.float32)

    if normalizer is not None:
        obs_batch = normalizer.normalize_obs(obs_batch)

    obs_tensor = torch.as_tensor(obs_batch, device=model.policy.device)

    with torch.no_grad():
        q_values = _extract_q_values(model, obs_tensor)  # (B, 3)

    q_np = q_values.cpu().numpy()  # (B, 3)
    q_std = np.std(q_np, axis=1)  # (B,)
    q_max = np.max(q_np, axis=1)  # (B,)
    q_mean = np.mean(q_np, axis=1)  # (B,)

    # Avoid division by zero
    safe_std = np.where(q_std < 1e-12, 1.0, q_std)
    spread = (q_max - q_mean) / safe_std
    spread = np.where(q_std < 1e-12, 0.0, spread)

    return spread


def load_executor_dqn(
    model_dir: str | Path,
) -> tuple[DQN, StaticNormalizer | None]:
    """Load a trained DQN Executor + optional StaticNormalizer.

    Args:
        model_dir: Directory containing ``model.zip`` and optionally
                   ``normalization_stats.json``.

    Returns:
        Tuple of (DQN model, StaticNormalizer or None).
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.zip"
    stats_path = model_dir / "normalization_stats.json"

    if not model_path.exists():
        raise FileNotFoundError(f"DQN model not found: {model_path}")

    model = DQN.load(str(model_path))

    normalizer: StaticNormalizer | None = None
    if stats_path.exists():
        normalizer = StaticNormalizer.load(stats_path)

    return model, normalizer
