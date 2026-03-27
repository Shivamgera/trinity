"""Select the best model from a W&B hyperparameter sweep.

This script queries the W&B API to find the run with the highest
``val/sharpe_ratio`` and copies its artifacts to:
    experiments/executor/best_model/

Usage:
    python3 -m src.executor.select_best --sweep-id <sweep_id>
    python3 -m src.executor.select_best --sweep-id <sweep_id> --entity myteam --project myproject
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

BEST_MODEL_DIR = Path("experiments/executor/best_model")
SWEEP_RUN_DIR = Path("experiments/executor/sweep")


def select_best_model(
    sweep_id: str,
    entity: str | None = None,
    project: str | None = None,
    metric: str = "val/sharpe_ratio",
    minimize: bool = False,
) -> dict:
    """Query W&B API to find and copy the best sweep run's model.

    Args:
        sweep_id: W&B sweep ID (e.g. "abc12345").
        entity: W&B entity (user or team). Uses default if None.
        project: W&B project name. Uses default if None.
        metric: Metric to optimize. Higher is better unless minimize=True.
        minimize: If True, select the run with the lowest metric value.

    Returns:
        Dict with best run info: run_id, metric_value, config, run_dir.
    """
    api = wandb.Api()

    # Build sweep path
    if entity and project:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    elif project:
        sweep_path = f"{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    logger.info(f"Querying sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    best_run = None
    best_value = float("inf") if minimize else float("-inf")

    for run in sweep.runs:
        if run.state != "finished":
            continue

        # Get the target metric
        summary = run.summary
        if metric not in summary:
            logger.warning(f"Run {run.id} missing metric '{metric}', skipping.")
            continue

        value = summary[metric]

        if minimize:
            if value < best_value:
                best_value = value
                best_run = run
        else:
            if value > best_value:
                best_value = value
                best_run = run

    if best_run is None:
        raise RuntimeError(
            f"No finished runs with metric '{metric}' found in sweep {sweep_path}"
        )

    logger.info(f"Best run: {best_run.id} | {metric}={best_value:.4f}")
    logger.info(f"Best config: {dict(best_run.config)}")

    # Copy model artifacts to best_model directory
    run_dir = SWEEP_RUN_DIR / best_run.id
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Local sweep run directory not found: {run_dir}\n"
            f"Make sure sweep was run locally with run IDs matching W&B run IDs."
        )

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy model files
    for filename in ("model.zip", "vec_normalize.pkl"):
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, BEST_MODEL_DIR / filename)
            logger.info(f"Copied {src} → {BEST_MODEL_DIR / filename}")
        else:
            logger.warning(f"File not found, skipping: {src}")

    # Save best run metadata
    metadata = {
        "sweep_id": sweep_id,
        "best_run_id": best_run.id,
        "metric": metric,
        "metric_value": float(best_value),
        "config": dict(best_run.config),
    }
    with open(BEST_MODEL_DIR / "best_run.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Best model saved to {BEST_MODEL_DIR}")
    return metadata


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Select best sweep model")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--project", default=None, help="W&B project")
    parser.add_argument(
        "--metric",
        default="val/sharpe_ratio",
        help="Metric to maximize",
    )
    args = parser.parse_args()

    metadata = select_best_model(
        sweep_id=args.sweep_id,
        entity=args.entity,
        project=args.project,
        metric=args.metric,
    )
    print(f"\nBest run: {metadata['best_run_id']}")
    print(f"  {args.metric}: {metadata['metric_value']:.4f}")
    print(f"  Config: {metadata['config']}")
    print(f"\nModel saved to: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()
