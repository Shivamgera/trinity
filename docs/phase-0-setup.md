# Phase 0: Environment & Project Scaffold

**Project:** Robust Trinity — Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Timeline:** Week 1, ~8-12 hours total
**Goal:** Initialize the project repository, install all dependencies, fork and modernize the ABIDES simulator, and verify the full development toolchain.

---

## P0-T1: Initialize Project Structure and Git Repository

**Estimated time:** ~2 hours
**Dependencies:** None (this is the first task)

### Context

We are building a multi-agent financial trading system called the "Robust Trinity." The system consists of three agents and a coordination mechanism:

1. **Analyst** (LLM-based): Processes financial text → outputs (decision, reasoning)
2. **Executor** (PPO-based): Processes numeric market data → outputs a policy distribution over {hold, buy, sell}
3. **Guardian** (Rule-based): Applies hard constraints and adaptive post-decision checks
4. **C-Gate** (Consistency Gate): Computes divergence Δ = 1 - π_RL(d_LLM) between Analyst decision and Executor policy

Key architectural invariants:
- Channel independence: Analyst and Executor share ZERO features
- Pre-trained frozen agents, no online learning
- Position-target action space: {flat=0, long=1, short=2}

The tech stack is: Python 3.11+, Stable-Baselines3, Gymnasium, anthropic SDK, Pydantic, W&B, pytest, ollama (local Llama for dev), ruff, yfinance, torch.

The project root is `/Users/shivamgera/projects/research1`.

### Objective

Create the full project directory structure, configure `pyproject.toml` with all dependencies, set up `.gitignore` and `.env.example`, create all Python package `__init__.py` files, and make the initial Git commit.

### Detailed Instructions

**Step 1: Create the directory structure**

Create the following directories and files under `/Users/shivamgera/projects/research1/`:

```
research1/
├── configs/
│   ├── abides/
│   ├── ppo/
│   └── llm/
├── src/
│   ├── analyst/
│   ├── executor/
│   ├── guardian/
│   ├── cgate/
│   ├── trinity/
│   ├── adversarial/
│   ├── evaluation/
│   └── utils/
├── data/
│   ├── raw/
│   ├── processed/
│   └── abides/
├── experiments/
├── notebooks/
├── tests/
├── pyproject.toml
├── .gitignore
└── .env.example
```

Every directory under `src/` must contain an `__init__.py` file (can be empty). The top-level `src/` must also have an `__init__.py`. The `configs/`, `data/`, `experiments/`, `notebooks/`, and `tests/` directories should each contain a `.gitkeep` file so Git tracks empty directories.

**Step 2: Create `pyproject.toml`**

Use the `[project]` table (PEP 621). The project name should be `robust-trinity`. Use `src` layout with `packages = ["src"]` or a flat layout — whichever is simpler. The key point is that `import src.executor` etc. must work.

```toml
[project]
name = "robust-trinity"
version = "0.1.0"
description = "Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection"
requires-python = ">=3.11"
dependencies = [
    "stable-baselines3>=2.3.0",
    "gymnasium>=0.29.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "scipy>=1.11.0",
    "anthropic>=0.39.0",
    "pydantic>=2.0",
    "wandb>=0.16.0",
    "pytest>=7.4.0",
    "ruff>=0.1.0",
    "yfinance>=0.2.30",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "torch>=2.0",
    "pyyaml>=6.0",
    "pyarrow>=14.0",
    "python-dotenv>=1.0.0",
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 3: Create `.gitignore`**

Include the following patterns:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Environment
.env
.venv/
venv/
env/

# Data and experiments (large files)
data/raw/
data/processed/
data/abides/
experiments/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# W&B
wandb/

# Jupyter
.ipynb_checkpoints/

# Model files
*.zip
*.pkl
*.pt
*.pth
```

**Step 4: Create `.env.example`**

```
ANTHROPIC_API_KEY=
WANDB_API_KEY=
WANDB_PROJECT=robust-trinity
OLLAMA_HOST=http://localhost:11434
```

**Step 5: Create all `__init__.py` files**

Create empty `__init__.py` files in:
- `src/__init__.py`
- `src/analyst/__init__.py`
- `src/executor/__init__.py`
- `src/guardian/__init__.py`
- `src/cgate/__init__.py`
- `src/trinity/__init__.py`
- `src/adversarial/__init__.py`
- `src/evaluation/__init__.py`
- `src/utils/__init__.py`
- `tests/__init__.py`

**Step 6: Initialize Git and make initial commit**

```bash
git init
git add .
git commit -m "P0-T1: initialize project structure and dependencies"
```

### Acceptance Criteria

1. Running `find . -name "__init__.py"` from project root lists 10 files (9 under `src/`, 1 under `tests/`)
2. Running `pip install -e .` from project root completes without errors
3. Running `python -c "import src; import src.executor; import src.analyst; import src.cgate"` succeeds
4. `git log --oneline` shows exactly one commit
5. `.gitignore` prevents `data/processed/`, `experiments/`, `.env`, `wandb/`, and `__pycache__/` from being tracked
6. `pyproject.toml` lists all 17 dependencies

### Files to Create

- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `src/__init__.py` and all 8 sub-package `__init__.py` files
- `tests/__init__.py`
- `.gitkeep` files in `configs/abides/`, `configs/ppo/`, `configs/llm/`, `data/raw/`, `data/processed/`, `data/abides/`, `experiments/`, `notebooks/`

### Human Checkpoint

- Verify `pip install -e .` completes successfully
- Verify all imports work
- Review `pyproject.toml` for correctness (especially dependency versions)
- Confirm the directory tree matches the specification

---

## P0-T2: Seed Management and Utility Setup

**Estimated time:** ~2 hours
**Dependencies:** P0-T1 must be completed first (project structure and dependencies must exist)

### Context

This is part of the "Robust Trinity" project — a multi-agent financial trading system for a master's thesis. The project structure has already been created (P0-T1) with all packages under `src/`. Dependencies are installed via `pyproject.toml`.

The system uses:
- **PyTorch** (for the PPO-based Executor agent via Stable-Baselines3)
- **NumPy** (for numeric computation throughout)
- **Gymnasium** (for the trading environment)
- **W&B** (Weights & Biases for experiment tracking)

Reproducibility is critical for a thesis project. Every experiment must be reproducible given the same seed. W&B is used to track all experiments with a consistent naming convention.

The project root is `/Users/shivamgera/projects/research1`. The `src/utils/` package already has an empty `__init__.py`.

### Objective

Create seed management utilities, a W&B logging wrapper, stub data loading functions, a base configuration file, and tests verifying reproducibility.

### Detailed Instructions

**Step 1: Create `src/utils/seed.py`**

```python
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
```

**Step 2: Create `src/utils/logging.py`**

```python
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
```

**Step 3: Create `src/utils/data.py`**

These are stub functions that will be fully implemented in Phase 1. They should have correct signatures and docstrings but raise `NotImplementedError` for now.

```python
"""Data loading and feature engineering utilities.

This module handles loading numeric market data and text headlines
for the Robust Trinity system. The Analyst and Executor agents use
completely independent data channels:
- Executor: numeric OHLCV-derived features (z-normalized)
- Analyst: text headlines (no numeric features)

This channel independence is a core architectural invariant.
"""

from pathlib import Path
from typing import Any

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_numeric_features(
    ticker: str = "AAPL",
    split: str = "train",
) -> pd.DataFrame:
    """Load z-normalized numeric features for the Executor agent.

    Args:
        ticker: Stock ticker symbol.
        split: One of "train", "val", "test".

    Returns:
        DataFrame with DatetimeIndex and ~15 numeric feature columns.
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T1)")


def load_headlines(
    ticker: str = "AAPL",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Load text headlines for the Analyst agent.

    Args:
        ticker: Stock ticker symbol.
        start_date: ISO format start date (inclusive).
        end_date: ISO format end date (inclusive).

    Returns:
        List of dicts with keys: date, ticker, headline, source.
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T2)")


def get_feature_names() -> list[str]:
    """Return the ordered list of numeric feature column names.

    Returns:
        List of feature name strings matching columns in load_numeric_features().
    """
    raise NotImplementedError("Will be implemented in Phase 1 (P1-T1)")


def verify_channel_independence(features_df: pd.DataFrame) -> None:
    """Assert that the numeric feature DataFrame contains no text-derived columns.

    This is a core architectural invariant: the Executor sees ONLY numeric
    market data, never any text or NLP features.

    Args:
        features_df: The numeric features DataFrame.

    Raises:
        AssertionError: If any column name suggests text-derived data.
    """
    text_indicators = ["sentiment", "text", "headline", "nlp", "embedding", "token"]
    for col in features_df.columns:
        for indicator in text_indicators:
            assert indicator not in col.lower(), (
                f"Channel independence violation: column '{col}' appears to be text-derived "
                f"(contains '{indicator}'). Executor must only see numeric market features."
            )
```

**Step 4: Create `configs/base.yaml`**

```yaml
# Robust Trinity — Base Configuration
# Project: Structural Fault Tolerance in Heterogeneous Financial AI Systems

project:
  name: "robust-trinity"
  version: "0.1.0"

seed: 42

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  data_abides: "data/abides"
  experiments: "experiments"

data:
  ticker: "AAPL"
  train_start: "2022-01-01"
  train_end: "2024-06-30"
  test_start: "2024-07-01"
  test_end: "2024-12-31"
  lookback_window: 30

wandb:
  project: "robust-trinity"
  entity: null  # set in .env or override

action_space:
  # Position-target action space
  # 0 = flat (no position), 1 = long, 2 = short
  flat: 0
  long: 1
  short: 2
```

**Step 5: Create `tests/test_utils.py`**

```python
"""Tests for utility modules — seed reproducibility and data stubs."""

import numpy as np
import torch
import random

from src.utils.seed import set_global_seed


class TestSeedReproducibility:
    """Verify that set_global_seed produces identical sequences."""

    def test_numpy_reproducibility(self):
        set_global_seed(42)
        a1 = np.random.rand(10)
        set_global_seed(42)
        a2 = np.random.rand(10)
        np.testing.assert_array_equal(a1, a2)

    def test_torch_reproducibility(self):
        set_global_seed(42)
        t1 = torch.randn(10)
        set_global_seed(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_random_reproducibility(self):
        set_global_seed(42)
        r1 = [random.random() for _ in range(10)]
        set_global_seed(42)
        r2 = [random.random() for _ in range(10)]
        assert r1 == r2

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a1 = np.random.rand(10)
        set_global_seed(99)
        a2 = np.random.rand(10)
        assert not np.array_equal(a1, a2)


class TestDataStubs:
    """Verify data loading stubs raise NotImplementedError."""

    def test_load_numeric_features_not_implemented(self):
        from src.utils.data import load_numeric_features
        try:
            load_numeric_features()
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

    def test_load_headlines_not_implemented(self):
        from src.utils.data import load_headlines
        try:
            load_headlines()
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass
```

**Step 6: Update `src/utils/__init__.py`**

```python
"""Utility modules for the Robust Trinity system."""

from src.utils.seed import set_global_seed
```

**Step 7: Run the tests and commit**

```bash
pytest tests/test_utils.py -v
git add .
git commit -m "P0-T2: add seed management, logging, data stubs, base config, and tests"
```

### Acceptance Criteria

1. `pytest tests/test_utils.py -v` passes all 5 tests
2. `from src.utils.seed import set_global_seed` works
3. `from src.utils.logging import init_wandb` works
4. `from src.utils.data import load_numeric_features, load_headlines, verify_channel_independence` works
5. `configs/base.yaml` is valid YAML and loads with `yaml.safe_load()`
6. Seed reproducibility test demonstrates identical random sequences with same seed and different sequences with different seeds

### Files to Create

- `src/utils/seed.py`
- `src/utils/logging.py`
- `src/utils/data.py`
- `configs/base.yaml`
- `tests/test_utils.py`

### Files to Modify

- `src/utils/__init__.py` (add imports)

### Human Checkpoint

- Run `pytest tests/test_utils.py -v` and confirm all tests pass
- Review `configs/base.yaml` for any missing or incorrect settings
- Confirm date ranges make sense for AAPL data

---

## P0-T3: Fork and Modernize ABIDES Simulator

**Estimated time:** ~3 hours
**Dependencies:** P0-T1 must be completed first (project structure and pyproject.toml must exist)

### Context

The Robust Trinity project uses the **ABIDES** (Agent-Based Interactive Discrete Event Simulation) simulator for evaluation of the multi-agent trading system. ABIDES was developed by JP Morgan Research and is available at `https://github.com/jpmorganchase/abides-jpmc-public`.

ABIDES simulates realistic market microstructure with multiple agent types (zero-intelligence, momentum, market makers, etc.). We will use the **RMSC04** configuration, which is a realistic multi-agent market simulation.

The problem: ABIDES was written for older dependency versions:
- `gym==0.18.0` (we need `gymnasium>=0.29.0`)
- `ray[rllib]==1.7.0` (we don't need Ray at all — we use Stable-Baselines3)
- Older numpy/pandas/scipy

The Gymnasium API changes are significant:
- `env.reset()` now returns `(observation, info)` instead of just `observation`
- `env.step(action)` now returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`

We need ABIDES as a local dependency for running market simulations during evaluation (later phases). It does NOT need to be perfect now — we just need it installable and the RMSC04 config running without crashing.

### Objective

Fork the ABIDES repository, update its dependencies to be compatible with our stack, fix the Gymnasium API changes, and verify that the RMSC04 simulation config runs to completion.

### Detailed Instructions

**Step 1: Clone ABIDES into the project**

```bash
cd /Users/shivamgera/projects/research1
git clone https://github.com/jpmorganchase/abides-jpmc-public.git abides
```

If the repository is no longer available or has moved, search for the latest fork. The key packages in ABIDES are:
- `abides-core`: The discrete event simulation engine
- `abides-markets`: Market-specific agents and order books
- `abides-gym`: Gymnasium wrappers for the simulator

**Step 2: Examine the current dependency structure**

Look at `abides/setup.py` or `abides/pyproject.toml` (and any sub-packages) to understand the current dependency pinning. Key files to check:
- `abides-core/setup.cfg` or `setup.py`
- `abides-markets/setup.cfg` or `setup.py`
- `abides-gym/setup.cfg` or `setup.py`

**Step 3: Update dependencies**

For each ABIDES sub-package, update:
- Replace `gym` with `gymnasium` (or add gymnasium compatibility)
- Remove `ray[rllib]` dependency entirely
- Update numpy, pandas, scipy to be compatible with Python 3.11+
- Fix any other incompatible version pins

Key search-and-replace patterns:
- `import gym` → `import gymnasium as gym`
- `from gym import ...` → `from gymnasium import ...`
- `gym.Env` → `gymnasium.Env`
- In `reset()` methods: if they return only `obs`, change to return `(obs, {})`
- In `step()` methods: if they return `(obs, reward, done, info)`, change to return `(obs, reward, done, False, info)` (using `done` as `terminated` and `False` as `truncated`)

**Step 4: Fix import paths and API changes**

The most critical changes:
1. In all `gymnasium.Env` subclasses in `abides-gym/`:
   - `reset()` must return `(observation, info_dict)`
   - `step()` must return `(observation, reward, terminated, truncated, info_dict)`
2. `gym.spaces` → `gymnasium.spaces` (usually works as-is if import is aliased)
3. Remove any `ray`-specific code paths or make them optional

**Step 5: Install ABIDES as local editable packages**

Add to your project's `pyproject.toml` or install directly:

```bash
pip install -e abides/abides-core
pip install -e abides/abides-markets
pip install -e abides/abides-gym
```

If there are installation errors, fix them iteratively. Common issues:
- Missing `setup.cfg` fields
- Incompatible Cython or C extensions
- Circular dependencies between abides sub-packages

**Step 6: Verify RMSC04 config runs**

Look for the RMSC04 config file (likely in `abides/abides-markets/abides_markets/configs/` or similar). Run a short simulation:

```python
# Save as scripts/verify_abides.py
"""Verify ABIDES RMSC04 simulation runs to completion."""

# The exact import path depends on the ABIDES structure.
# Common patterns:
# from abides_markets.configs.rmsc04 import build_config
# OR find the relevant config builder

# Run a short simulation (e.g., 1 trading day, reduced number of agents)
# The goal is just to verify it doesn't crash, not to get realistic results.

print("ABIDES RMSC04 simulation completed successfully!")
```

The exact code will depend on ABIDES's structure. Look at the existing examples in `abides/` for how to run a simulation.

**Step 7: Add ABIDES to .gitignore and document**

Since ABIDES is a large external dependency, add it to `.gitignore`:
```
# ABIDES (installed as local editable, tracked separately)
abides/
```

Create a small note in `configs/abides/README.md` documenting:
- Which ABIDES version/commit was forked
- What changes were made
- How to install it

**Step 8: Commit**

```bash
git add .
git commit -m "P0-T3: fork and modernize ABIDES simulator for gymnasium compatibility"
```

### Acceptance Criteria

1. `pip install -e abides/abides-core` completes without errors
2. `pip install -e abides/abides-markets` completes without errors
3. `python -c "import abides_core"` succeeds
4. `python -c "import abides_markets"` succeeds
5. A short RMSC04 simulation runs to completion without crashing (even if the output isn't validated for correctness)
6. No `gym` (old) imports remain — all converted to `gymnasium`
7. `configs/abides/README.md` documents the fork and changes

### Files to Create

- `abides/` directory (cloned from GitHub, then modified)
- `configs/abides/README.md`
- `scripts/verify_abides.py` (optional verification script)

### Files to Modify

- `.gitignore` (add `abides/` exclusion)
- Multiple files within `abides/` (dependency updates, API migration)

### Human Checkpoint

- **CRITICAL:** Run the ABIDES verification script and confirm the simulation completes
- Review the console output for any warnings that might indicate deeper issues
- Verify that `import gymnasium` is used throughout (not old `import gym`)
- Check that the ABIDES fork is at a known commit hash (document this)

---

## P0-T4: Verify Development Toolchain

**Estimated time:** ~1.5 hours
**Dependencies:** P0-T1 and P0-T2 must be completed (project structure, dependencies, seed utilities, and tests must exist)

### Context

The Robust Trinity project uses several tools that must all work correctly before we proceed to building agents:

1. **Ollama with Llama 3.2 8B** — used during development as a local stand-in for the Anthropic API. The Analyst agent will use Claude in production but Llama locally to avoid API costs during development. Ollama runs a local inference server at `http://localhost:11434`.

2. **Stable-Baselines3 (SB3)** — the reinforcement learning library used for the Executor agent (PPO algorithm). Must verify that PPO training works end-to-end on a simple environment.

3. **Weights & Biases (W&B)** — experiment tracking. Every training run, sweep, and evaluation will be logged here. Must verify that metrics can be logged and viewed.

4. **pytest** — test framework. Must verify it discovers and runs our existing tests.

5. **ruff** — Python linter. Must pass on our current codebase.

### Objective

Verify that each tool in the development stack is installed, configured, and functional. Create small verification scripts where needed. Ensure the test suite passes and the linter is clean.

### Detailed Instructions

**Step 1: Verify Ollama and Llama model**

Check if ollama is installed:
```bash
ollama --version
```

If not installed, the human needs to install it from https://ollama.ai. Do NOT attempt to install it programmatically.

Pull the Llama model:
```bash
ollama pull llama3.2:latest
```

Note: The task description says `llama3.2:8b` but the actual tag may vary. Use `ollama list` to verify available models. The 8B parameter model is the target.

Create a quick verification:
```python
# scripts/verify_ollama.py
"""Verify ollama is running and Llama model responds."""

import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": "In one sentence, what is a stock market?",
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Response: {result.get('response', 'NO RESPONSE')[:200]}")
    print("Ollama verification: PASSED")

if __name__ == "__main__":
    test_ollama()
```

Run: `python scripts/verify_ollama.py`

**Step 2: Verify Stable-Baselines3 with a dummy PPO run**

```python
# scripts/verify_sb3.py
"""Verify SB3 PPO training works on CartPole."""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.utils.seed import set_global_seed

def test_sb3():
    set_global_seed(42)

    # Create vectorized environment
    env = make_vec_env("CartPole-v1", n_envs=2, seed=42)

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        seed=42,
    )

    # Train for 1000 steps
    model.learn(total_timesteps=1000)

    # Evaluate
    eval_env = gym.make("CartPole-v1")
    obs, info = eval_env.reset(seed=42)
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Evaluation reward: {total_reward}")
    print(f"SB3 PPO training: PASSED (reward={total_reward})")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    test_sb3()
```

Run: `python scripts/verify_sb3.py`

The reward doesn't need to be high (it's only 1000 training steps). We just need to confirm the training loop runs without errors.

**Step 3: Verify W&B logging**

```python
# scripts/verify_wandb.py
"""Verify W&B logging works."""

import wandb
from src.utils.logging import init_wandb, log_metrics, finish_wandb

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
```

Run: `python scripts/verify_wandb.py`

This will require the user to have authenticated with W&B (`wandb login`) or have `WANDB_API_KEY` in `.env`. If offline mode is preferred, set `WANDB_MODE=offline` in the environment.

**Step 4: Run pytest**

```bash
pytest tests/test_utils.py -v
```

All 5 tests from P0-T2 should pass. Verify the output shows:
- `test_numpy_reproducibility PASSED`
- `test_torch_reproducibility PASSED`
- `test_random_reproducibility PASSED`
- `test_different_seeds_differ PASSED`
- `test_load_numeric_features_not_implemented PASSED`
- `test_load_headlines_not_implemented PASSED`

**Step 5: Run ruff linter**

```bash
ruff check src/ tests/
```

Fix any issues found. Common issues:
- Unused imports
- Missing whitespace
- Line too long (we set 100 char limit in pyproject.toml)

If there are issues, fix them and re-run until clean.

**Step 6: Create a verification summary and commit**

Create a file `scripts/README.md`:
```markdown
# Verification Scripts

These scripts verify the development toolchain. Run them in order:

1. `python scripts/verify_ollama.py` — Tests local Llama model via ollama
2. `python scripts/verify_sb3.py` — Tests PPO training via Stable-Baselines3
3. `python scripts/verify_wandb.py` — Tests experiment logging via W&B
```

```bash
git add .
git commit -m "P0-T4: verify development toolchain (ollama, SB3, W&B, pytest, ruff)"
```

### Acceptance Criteria

1. `ollama list` shows a Llama 3.2 model
2. `verify_ollama.py` gets a coherent text response
3. `verify_sb3.py` completes PPO training on CartPole without errors
4. `verify_wandb.py` creates a W&B run and logs 10 steps of metrics
5. `pytest tests/ -v` passes all tests
6. `ruff check src/ tests/` reports 0 errors
7. All verification scripts exist in `scripts/`

### Files to Create

- `scripts/verify_ollama.py`
- `scripts/verify_sb3.py`
- `scripts/verify_wandb.py`
- `scripts/README.md`

### Files to Modify

- Potentially any `src/` files if ruff finds issues

### Human Checkpoint

- Verify ollama is running locally (`ollama serve` must be active)
- Check the W&B dashboard to confirm the verification run appears
- Confirm no dependency conflicts between SB3, gymnasium, and torch
- If any verification fails, debug and fix before proceeding to Phase 1
