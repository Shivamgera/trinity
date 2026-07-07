# Robust Trinity — Cross-Modal Arbitration for Structural Fault Containment

Reference implementation and reproduction package for the paper:

> **Cross-Modal Arbitration for Structural Fault Containment Under Adversarial Signal Injection**
> Shivam Gera, Ali Beikmohammadi, Alfreds Lapkovskis.
> Department of Computer and Systems Sciences, Stockholm University.

This repository lets a reviewer reproduce every quantitative claim in the
paper: the clean baselines (Table 5), the adversarial robustness ranges
(Table 6), the C-Gate ablation (Table 7), the block-bootstrap statistical
inference (Section 4.4), and all figures.

> **Note to reviewers.** The trained artifacts required to reproduce the
> paper tables (frozen PPO checkpoints, precomputed GPT-5 signals, and the
> processed feature/headline files) are versioned under `experiments/` and
> `data/processed/`. You can therefore reproduce **all reported numbers
> without any API keys or GPU** by running the evaluation stage directly
> (Steps 5–8 below). Steps 2–4 (data download, executor training, signal
> precomputation) are provided for full end-to-end reproduction and require
> external API keys.

---

## 1. What this repository reproduces

The system pairs two channels that share **zero** input features:

| Component | Channel | Model | Input | Output |
|-----------|---------|-------|-------|--------|
| **Analyst** | A (semantic) | GPT-5, frozen, precomputed | Text headline | Directional signal `d_A ∈ {hold, buy, sell}` |
| **Executor** | B (numeric) | PPO (Stable-Baselines3), frozen | 143-dim numeric obs | Policy distribution `π_B` over `{hold, buy, sell}` |
| **C-Gate** | arbitration | Deterministic | `d_A`, `π_B` | Divergence `Δ_t = 1 − π_B(d_A ∣ o_t)` → regime |
| **Guardian** | safety | Rule-based | Signals + portfolio state | Hard constraints + regime-dependent exposure |

**Locked architectural constants** (must match the paper — do not change):

- C-Gate metric: `Δ_t = 1 − π_B(d_A | o_t)` (asymmetric; **not** JSD).
- Channel independence: Analyst sees text only; Executor sees numerics only.
- Three-regime policy: Agreement (`Δ ≤ τ_low`) → full position; Ambiguity
  (`τ_low < Δ < τ_high`) → 50% position + 2% stop-loss; Conflict
  (`Δ ≥ τ_high`) → flat.
- Action space `{flat=0, long=1, short=2}`, single ticker AAPL, daily bars.
- Observation: 14 z-normalised features × lookback 10 + 3 portfolio state = 143 dims.
- C-Gate final config: `T=1.0`, `τ_low=0.6329` (P20), `τ_high=0.6991` (P80),
  GPT-5 signals (`data/processed/precomputed_signals_gpt5.json`).
- Frozen Executor seeds: **1111, 4096, 999, 9999**.
- Test window: 2024-07-01 → 2024-12-30 (127 trading days, 116 decision days).
- Reported evaluation metrics: total return, Sharpe, MaxDD. (Sortino is the
  training reward only, not an evaluation metric.)

---

## 2. Claim → artifact map (for reviewers)

| Paper element | Produced by | Reads from |
|---------------|-------------|------------|
| Table 5 (clean baselines) | `scripts/run_baselines.py`, `scripts/calibrate_and_run.py` | frozen checkpoints, GPT-5 signals |
| Table 6 (adversarial MaxDD ranges) | `scripts/run_adversarial.py --attack all` | frozen checkpoints, GPT-5 signals |
| Table 7 (C-Gate ablation) | `scripts/run_baselines.py` + `scripts/calibrate_and_run.py` | `experiments/` JSONs |
| Section 4.4 (bootstrap, CIs, p-values) | `scripts/bootstrap_analysis.py` | per-day result JSONs in `experiments/` |
| Figures (Δ histogram, MaxDD curves) | `scripts/generate_figures.py` | `experiments/` JSONs |
| Threshold calibration (τ_low, τ_high) | `scripts/calibrate_and_run.py --calibrate-only` | GPT-5 signals, val split |

---

## 3. System requirements

- **OS:** macOS or Linux (developed on macOS M-series and a Linux SLURM cluster).
- **Python:** ≥ 3.11.
- **Hardware:** CPU-only is sufficient for evaluation and single-seed PPO
  training. No GPU is required to reproduce the reported tables.
- **Disk:** ~2 GB including artifacts.
- **Compute budget (full end-to-end):** the reported results were produced on
  4 frozen seeds; the executor hyperparameter search used a 30-run Bayesian
  sweep. API cost for GPT-5 signal precomputation is bounded by 1,021 headlines.

---

## 4. Installation

```bash
git clone <this-repo-url> research1
cd research1

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install --upgrade pip
pip install -e .
```

`pip install -e .` installs the pinned dependency set from
[pyproject.toml](pyproject.toml) (Stable-Baselines3, Gymnasium, PyTorch,
NumPy/Pandas/SciPy, yfinance, matplotlib, etc.) and exposes the `src` package
on the path used by the scripts.

**Optional — ABIDES simulator.** The market-simulation components under
`abides/` (a modernised fork of `abides-jpmc-public`) are only needed for the
simulated-market extensions and are **not** required for the paper tables:

```bash
cd abides && ./install.sh && cd ..
```

**Optional — API keys.** Only required for Steps 5–6 (data download and
signal precomputation). Copy the template and fill in what you need:

```bash
cp .env.example .env
```

```dotenv
# .env
POLYGON_API_KEY=      # headlines (Step 5b); free tier, 5 req/min
AZURE_OPENAI_API_KEY= # GPT-5 Analyst signals (Step 6)
ANTHROPIC_API_KEY=    # optional alternative Analyst backend (Claude)
OLLAMA_HOST=http://localhost:11434  # optional local Llama backend
WANDB_API_KEY=        # optional; only for training sweeps
WANDB_PROJECT=robust-trinity
```

---

## 5. Fast path — reproduce the paper tables (no API keys, no GPU)

The frozen checkpoints and processed data are already in the repository, so
reviewers can jump straight to evaluation.

```bash
# 5a. Calibrate C-Gate thresholds on validation, then run Trinity on test.
#     Reproduces τ_low=0.6329, τ_high=0.6991 and the Trinity row of Table 5.
python scripts/calibrate_and_run.py

# 5b. Run all four ablation baselines on the test split (Table 5 / Table 7).
python scripts/run_baselines.py --baseline all --split both

# 5c. Run all three attack families at five intensities each (Table 6).
python scripts/run_adversarial.py --attack all --split test

# 5d. Block-bootstrap CIs and paired significance tests (Section 4.4).
python scripts/bootstrap_analysis.py \
    --results-root experiments \
    --seeds 999 1111 4096 9999 \
    --block-length 10 \
    --n-bootstrap 10000 \
    --output experiments/bootstrap/bootstrap_summary.json

# 5e. Regenerate all thesis figures from the JSON artifacts.
python scripts/generate_figures.py --output-dir thesis_text/figures
```

All outputs are written as JSON under `experiments/` (one file per
configuration/seed/attack), which the figure and bootstrap scripts consume.

---

## 6. Full end-to-end reproduction (from raw data)

Only needed if you want to rebuild artifacts from scratch. Requires the API
keys from Step 4.

**6a. Numeric data + features (Yahoo Finance).**
```bash
python -m scripts.download_data                # AAPL only (default)
python -m scripts.download_data --multi-ticker # + MSFT/GOOGL/SPY/AMZN augmentation
```
Produces `data/processed/aapl_features.parquet` (14 z-normalised features,
OHLCV 2007–2024).

**6b. Text data (Polygon.io headlines).**
```bash
python scripts/download_headlines.py           # AAPL headlines 2020–2024
```
Produces `data/processed/headlines.json` (1,021 headlines, one per trading day).

**6c. Analyst signal precomputation (GPT-5).**
```bash
python scripts/run_precompute.py               # → precomputed_signals_gpt5.json
# alternatives: --backend ollama (local, free) | --backend claude
```
GPT-5 runs at temperature 0 (deterministic). Signals are precomputed **offline**
and only enter the system at C-Gate evaluation time, so the absence of
pre-2020 headlines does not affect executor training.

**6d. Executor (PPO) training across frozen seeds.**
```bash
python -m scripts.run_multiseed                     # train + evaluate all seeds
python -m scripts.run_multiseed --eval-only --freeze # evaluate + freeze top 4
```
Frozen checkpoints are written to
`experiments/executor/frozen/seed_<seed>/{model.zip, vec_normalize.pkl}`.
Locked hyperparameters (γ=0.97, Sortino training reward, 2×64 ReLU MLP) are
defined in [scripts/run_multiseed.py](scripts/run_multiseed.py).

**6e. C-Gate threshold sweep (optional).**
```bash
python scripts/sweep_cgate.py --top 20   # 125-config grid over T, τ_low, τ_high
```
The paper's percentile calibration (P20/P80, T=1.0) was selected from this
sweep for stability on unseen test data.

Then run the evaluation stage (Step 5) to regenerate tables and figures.

---

## 7. Expected results

Reproduced numbers should match the paper within floating-point tolerance
(evaluation is deterministic given the frozen artifacts and fixed corruption
seed 42).

**Clean test-split performance (paper Table 5):**

| Configuration | Sharpe | Return (%) | MaxDD (%) |
|---------------|-------:|-----------:|----------:|
| Buy-and-Hold | +1.193 | +10.44 | 9.46 |
| Analyst-Only | −0.068 | −0.83 | 9.12 |
| Executor-Only | +1.753 | +9.70 | 4.51 |
| Trinity-no-CGate | +1.781 | +5.64 | 3.57 |
| Trinity | +1.561 | +3.57 | **3.17** |

**Adversarial MaxDD ranges across 15 scenarios (paper Table 6):**

| Attack | Trinity | Executor-Only | Trinity-no-CGate |
|--------|---------|---------------|------------------|
| Semantic poisoning | 2.50–3.53 | 4.51 (invariant) | 3.15–3.67 |
| Numeric perturbation | 3.17–4.03 | 4.29–4.95 | 3.50–3.57 |
| Action-flipping | 3.24–7.07 | 5.14–11.01 | 4.16–6.95 |
| **All 15 scenarios** | **2.50–7.07** | 4.29–11.01 | 3.15–6.95 |

**Statistical inference (paper Section 4.4):** Trinity vs. Trinity-no-CGate
MaxDD difference −0.73 pp (p=0.083, 95% CI [−1.67, +0.11]); Trinity vs.
Executor-Only MaxDD difference −4.88 pp (p=0.007, 95% CI [−10.85, −1.17]).

---

## 8. Repository structure

```
src/
  analyst/      Channel A: LLM client, backends (GPT-5/Claude/Ollama), signal precompute
  executor/     Channel B: Gymnasium TradingEnv, PPO training, policy distribution interface
  cgate/        Divergence metric Δ, threshold calibration, three-regime ConsistencyGate
  guardian/     Hard constraints (Stage 1) + regime-dependent exposure scaling (Stage 2)
  trinity/      End-to-end pipeline orchestration
  evaluation/   Metrics (Sharpe, Sortino, MaxDD), backtester
  utils/        Feature engineering, seeding, data loaders
scripts/        Reproduction entry points (download, train, evaluate, adversarial, figures)
configs/        base.yaml, data_splits.yaml, guardian.yaml, sweep configs
data/processed/ Feature parquets, headlines, precomputed GPT-5 signals
experiments/    Frozen checkpoints + per-config/seed/attack result JSONs
docs/           Phase-by-phase implementation notes and experiment-log.md
tests/          Unit tests for every component
abides/         Modernised ABIDES fork (optional market simulation)
```

Authoritative sources of truth for facts and results:
- [docs/experiment-log.md](docs/experiment-log.md) — all experiment results and decisions.
- [configs/data_splits.yaml](configs/data_splits.yaml) — exact date splits.
- [data/processed/precomputed_signals_gpt5.json](data/processed/precomputed_signals_gpt5.json) — 1,021 Analyst signals.

---

## 9. Configuration and determinism

- Global seed is set via `configs/base.yaml` (`seed: 42`) and
  `src/utils/seed.py`; adversarial corruption uses a fixed seed (42) for
  reproducible signal flipping and noise injection.
- Data splits are chronological and non-overlapping (training 2010–2023,
  validation Jan–Jun 2024, test Jul–Dec 2024); the test split is held out for
  all reported results.
- The Executor is deployed with **frozen** parameters (`model.zip` +
  `vec_normalize.pkl`); no updates occur at inference time.
- The Analyst is deterministic (temperature 0) and precomputed, so evaluation
  does not call any LLM at run time.

---

## 10. Tests

```bash
pytest                      # full suite
pytest tests/test_cgate.py  # C-Gate divergence + regime logic
```

Unit tests cover feature engineering, the trading environment, policy
extraction, the C-Gate divergence and calibration, the Guardian constraints,
and the integration pipeline.

---

## 11. Troubleshooting

- **`ModuleNotFoundError: src`** — run scripts from the repository root, or
  ensure `pip install -e .` completed (it registers the `src` package path).
- **Missing frozen checkpoints** — verify
  `experiments/executor/frozen/seed_<seed>/model.zip` exists; if not, run
  Step 6d, or use the checkpoints shipped in the repository.
- **API errors during Step 6** — Steps 5 (evaluation) require **no** API keys;
  only Steps 6a–6c contact external services.
- **Rate limits (Polygon free tier)** — the headline downloader sleeps 12.5 s
  between requests (5 req/min); a full crawl takes time by design.

---

## 12. Citation

```bibtex
@inproceedings{gera_crossmodal_trinity,
  title     = {Cross-Modal Arbitration for Structural Fault Containment
               Under Adversarial Signal Injection},
  author    = {Gera, Shivam and Beikmohammadi, Ali and Lapkovskis, Alfreds},
  institution = {Department of Computer and Systems Sciences,
                 Stockholm University},
  year      = {2026}
}
```

## 13. Acknowledgements

The computations were enabled by resources provided by the National Academic
Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the
Swedish Research Council through grant agreement no. 2022-06725.