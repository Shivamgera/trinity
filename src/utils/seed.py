"""Global seed management for reproducibility."""

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for all libraries to ensure reproducibility.

    Sets seeds for: random, numpy, torch (CPU and CUDA), gymnasium.
    Also configures torch for deterministic behavior.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gymnasium_seed(base_seed: int, env_index: int = 0) -> int:
    """Generate a unique seed for a gymnasium environment.

    Useful when running multiple vectorized environments.

    Args:
        base_seed: The base project seed.
        env_index: Index of the environment in a vectorized setup.

    Returns:
        A deterministic seed derived from base_seed and env_index.
    """
    return base_seed + env_index
