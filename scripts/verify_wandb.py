"""Verify W&B logging works."""

from src.utils.logging import finish_wandb, init_wandb, log_metrics


def test_wandb():
    run = init_wandb(
        phase="P0",
        component="verification",
        config={"test": True, "seed": 42},
        tags=["verification", "P0"],
    )

    # Log some dummy metrics
    for step in range(10):
        log_metrics(
            {
                "dummy/loss": 1.0 / (step + 1),
                "dummy/accuracy": step * 0.1,
            },
            step=step,
        )

    finish_wandb()
    print(f"W&B run URL: {run.url}")
    print("W&B verification: PASSED")


if __name__ == "__main__":
    test_wandb()
