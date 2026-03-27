"""Policy distribution extraction utilities for the Executor agent.

These functions allow downstream components (primarily the C-Gate, Phase 4)
to query the probability distribution over actions given an observation,
without needing to understand SB3 internals.

Usage:
    model, vec_normalize = load_executor("experiments/executor/best_model")
    probs = get_policy_distribution(model, obs, vec_normalize)
    # probs.shape == (3,) — [p_flat, p_long, p_short]
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def _extract_logits(model: PPO, obs_tensor: torch.Tensor) -> torch.Tensor:
    """Extract raw action logits from the policy network.

    Args:
        model: Trained PPO model.
        obs_tensor: Observation tensor of shape (B, obs_dim).

    Returns:
        Logits tensor of shape (B, n_actions).
    """
    features = model.policy.extract_features(
        obs_tensor, model.policy.features_extractor
    )
    latent_pi, _ = model.policy.mlp_extractor(features)
    return model.policy.action_net(latent_pi)


def get_policy_distribution(
    model: PPO,
    obs: np.ndarray,
    vec_normalize: VecNormalize | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Get the action probability distribution for a single observation.

    When the trained policy produces near-uniform softmax outputs (common
    with small networks on noisy financial data), lowering the temperature
    sharpens the distribution by dividing the raw logits by ``temperature``
    before applying softmax.  This is a standard post-hoc calibration
    technique that preserves the argmax ordering.

    Args:
        model: Trained PPO model.
        obs: Raw (un-normalized) observation of shape (obs_dim,) or (1, obs_dim).
        vec_normalize: Optional VecNormalize wrapper. If provided, obs is
                       normalized using its running statistics before being
                       passed to the policy.
        temperature: Softmax temperature (default 1.0 = no scaling).
                     Values < 1.0 sharpen the distribution; values > 1.0
                     flatten it.  Must be > 0.

    Returns:
        ndarray of shape (3,) with probabilities [p_flat, p_long, p_short].
        Sums to 1.0.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Ensure 2D: (1, obs_dim)
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]

    obs = obs.astype(np.float32)

    # Normalize if VecNormalize is present
    if vec_normalize is not None:
        obs = vec_normalize.normalize_obs(obs)

    # Convert to tensor
    obs_tensor = torch.as_tensor(obs, device=model.policy.device)

    # Extract logits and apply temperature-scaled softmax
    with torch.no_grad():
        logits = _extract_logits(model, obs_tensor)  # (1, 3)
        probs = torch.softmax(logits / temperature, dim=-1)  # (1, 3)

    return probs.cpu().numpy().flatten()  # shape (3,)


def get_policy_distribution_batch(
    model: PPO,
    obs_batch: np.ndarray,
    vec_normalize: VecNormalize | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Get action probability distributions for a batch of observations.

    Args:
        model: Trained PPO model.
        obs_batch: Raw observations of shape (B, obs_dim).
        vec_normalize: Optional VecNormalize for normalization.
        temperature: Softmax temperature (default 1.0 = no scaling).
                     Values < 1.0 sharpen the distribution.  Must be > 0.

    Returns:
        ndarray of shape (B, 3) with probabilities [p_flat, p_long, p_short].
        Each row sums to 1.0.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    obs_batch = obs_batch.astype(np.float32)

    if vec_normalize is not None:
        obs_batch = vec_normalize.normalize_obs(obs_batch)

    obs_tensor = torch.as_tensor(obs_batch, device=model.policy.device)

    with torch.no_grad():
        logits = _extract_logits(model, obs_tensor)  # (B, 3)
        probs = torch.softmax(logits / temperature, dim=-1)  # (B, 3)

    return probs.cpu().numpy()  # shape (B, 3)


def load_executor(
    model_dir: str | Path,
) -> tuple[PPO, VecNormalize | None]:
    """Load a trained Executor (PPO model + optional VecNormalize).

    Args:
        model_dir: Directory containing ``model.zip`` and optionally
                   ``vec_normalize.pkl``.

    Returns:
        Tuple of (PPO model, VecNormalize or None).
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.zip"
    vn_path = model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path))

    vec_normalize: VecNormalize | None = None
    if vn_path.exists():
        # Load using pickle directly to avoid SB3's set_venv(None) error.
        # We only need obs_rms/ret_rms for normalizing observations during
        # inference — we never call step/reset on this object directly.
        with open(vn_path, "rb") as f:
            vec_normalize = pickle.load(f)
        # Set to inference mode: no normalizing reward, no updating statistics
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    return model, vec_normalize
