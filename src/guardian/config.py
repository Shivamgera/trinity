"""Guardian configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.guardian.adaptive_policy import AdaptiveConfig
from src.guardian.hard_constraints import GuardianConfig


def load_guardian_config(
    config_path: str = "configs/guardian.yaml",
) -> tuple[GuardianConfig, AdaptiveConfig]:
    """Load Guardian configuration from a YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Tuple of (GuardianConfig, AdaptiveConfig).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Guardian config not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    hard_config = GuardianConfig(**raw.get("hard_constraints", {}))
    adaptive_config = AdaptiveConfig(**raw.get("adaptive_policy", {}))

    return hard_config, adaptive_config
