"""W&B experiment logging utilities."""

from datetime import datetime
from typing import Any

import wandb


def init_wandb(
    phase: str,
    component: str,
    config: dict[str, Any] | None = None,
    project: str = "robust-trinity",
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Initialize a W&B run with consistent naming convention.

    Run names follow the format: {phase}_{component}_{YYYYMMDD_HHMMSS}
    Example: "P2_executor_20250306_143022"

    Args:
        phase: Phase identifier (e.g., "P0", "P1", "P2").
        component: Component name (e.g., "executor", "analyst", "cgate").
        config: Dictionary of hyperparameters/config to log.
        project: W&B project name.
        tags: Optional list of tags for the run.

    Returns:
        The initialized W&B Run object.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{phase}_{component}_{timestamp}"

    run = wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        tags=tags or [phase, component],
        reinit=True,
    )
    return run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active W&B run.

    Args:
        metrics: Dictionary of metric_name -> value.
        step: Optional global step number.
    """
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish the current W&B run."""
    wandb.finish()
