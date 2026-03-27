# Workshop Paper Improvement Plan (NeurIPS/ICML AI in Finance)

## Timeline: 3-4 weeks

## Strategic Framing

The contribution is **architectural robustness via the C-Gate**, NOT alpha generation.
The paper story: "Multi-agent coordination through consistency gating reduces
portfolio drawdown by 40-55% under adversarial conditions, across multiple assets."

Negative mean Sharpe is acceptable at workshops IF:
- Framed as risk management (MaxDD reduction is the metric)
- Multi-asset results show the pattern generalizes
- Statistical tests confirm significance of MaxDD differences
- Ablations prove each component contributes

---

## Week 1: Multi-Asset Expansion + PPO Improvements

### 1.1 Add 4 more tickers (MSFT, GOOGL, JPM, SPY) — ~1 day

The codebase is already ticker-parameterized. Needed:
- Fix 2 hardcoded paths in `download_data.py` (use f-string with ticker)
- Fix hardcoded `TICKER="AAPL"` in `download_headlines.py` (CLI arg)
- Download OHLCV + compute features for each ticker (yfinance, ~1 min each)
- Download headlines from Polygon.io (~20 min per ticker)

**Ticker selection rationale:**
- MSFT: mega-cap tech peer (validates within-sector)
- GOOGL: mega-cap tech, different volatility profile
- JPM: different sector (finance), different return dynamics
- SPY: market index, tests whether architecture works on low-alpha asset

### 1.2 Precompute GPT-5 signals for new tickers — ~2-3 hours

Run `scripts/precompute.py` for each new ticker. ~500-1000 headlines per ticker.
No budget constraints per user.

### 1.3 Improve PPO training — ~2 days

The core problem: 746 training days for a 14,000-parameter network.
Two high-impact, low-effort changes:

**a) Reduce model capacity:**
- Try [16,16] and [32] architectures (vs current [64,64])
- Try linear policy (no hidden layers)
- A smaller model with 746 samples has much less overfitting risk

**b) Feature noise augmentation:**
- Add Gaussian noise N(0, sigma) to z-normalized features during training
- sigma in [0.01, 0.05, 0.1] — sweep this
- Implement in `_get_observation()` with a training-mode flag
- This effectively multiplies the dataset size

**c) Reduce feature set (optional):**
- Drop redundant features (bb_upper, bb_lower redundant with bb_percent_b;
  macd_line redundant with macd_histogram)
- Target: 7 features → obs_dim 73 → fewer parameters

### 1.4 Train PPO on all 5 tickers — ~1 day

- 20 seeds per ticker, sweep_train.py
- Both [64,64] and [16,16] architectures
- With and without noise augmentation
- ~15 min per ticker per config = ~2.5 hours total
- Select 4 best seeds per ticker using existing filter (entropy < 0.90 + combined score)

### 1.5 Run full pipeline per ticker — ~1 day

For each of 5 tickers:
- Calibrate C-Gate (T=1.0, p30/p80)
- Run Trinity integration (val + test)
- Run baselines (Executor-Only, Analyst-Only, Trinity-no-CGate)

---

## Week 2: Statistical Tests + Ablations + Additional Attacks

### 2.1 Implement statistical testing script — ~1 day

Create `scripts/statistical_tests.py` (spec exists in `docs/phase-8-evaluation.md`):

**Tier 1 (must-have):**
- Bootstrap Sharpe CI (10,000 resamples of ~117 daily returns per system)
- Bootstrap CI on Sharpe DIFFERENCE (Trinity minus each baseline)
- Paired t-test: Trinity daily returns vs B&H (with Newey-West HAC correction)
- Lo (2002) analytical Sharpe ratio SE: `SE = sqrt((1 + 0.5*SR^2) / n)`

**Tier 2 (important):**
- Permutation test for MaxDD difference (Trinity vs Executor-Only)
- Bootstrap CI on calibrated thresholds (tau_low, tau_high)
- Cohen's d effect size for all pairwise comparisons
- Holm-Bonferroni correction across multiple comparisons

Run across all 5 tickers × 4 seeds = 20 seed-ticker combinations.

### 2.2 Fix transaction costs in evaluation scripts — ~0.5 day

Critical gap: `cgate_integration.py`, `run_adversarial.py`, `run_baselines.py`
all compute cost-free returns. The env already computes costs internally.

Fix: Use `info["portfolio_return"]` (cost-adjusted) instead of computing
a separate `step_return = position * price_return` without costs.

This is a simple fix but changes ALL reported numbers. Must re-run everything.

Note: Current 15 bps (10 commission + 5 slippage) is actually conservative
for AAPL (real cost ~3-5 bps). This is GOOD for the paper — "conservative
cost assumptions" strengthens credibility. Keep 15 bps.

### 2.3 Implement Fused-LLM-RL baseline — ~1 day

This is the critical ablation for the channel independence claim.
Fused-LLM-RL: feed analyst signal directly into the RL observation space
(breaking channel independence). Compare performance to show that
separation + C-Gate is better than naive fusion.

### 2.4 Implement semantic attacks — ~1 day

Already designed in Phase 7/8 docs:
- Fabricated headlines (generate fake bullish/bearish headlines)
- Subtle manipulation (inject sentiment-shifting words into real headlines)
- More realistic than directional flips

### 2.5 Re-run adversarial evaluations for all tickers — ~1 day

- Analyst poisoning (5 rates) × 5 tickers × 4 seeds
- Executor perturbation (5 σ levels) × 5 tickers × 4 seeds
- Semantic attacks × 5 tickers × 4 seeds
- Compute all with transaction costs included

---

## Week 3: Paper Writing + Figures

### 3.1 Generate publication-ready figures — ~1 day

Key figures (6-8 for a 6-page workshop paper):
1. **Architecture diagram** — Trinity system with C-Gate, Analyst, Executor, Guardian
2. **MaxDD degradation curves** — MaxDD vs corruption rate/σ level, per system
3. **Regime distribution bar chart** — Agreement/Ambiguity/Conflict per ticker
4. **Δ distribution histogram** — Showing continuous spread at T=1.0
5. **Cross-asset MaxDD comparison** — Grouped bar chart, 5 tickers × 4 systems
6. **Bootstrap CI plots** — Sharpe and MaxDD with error bars per system

### 3.2 Write paper draft — ~3 days

Target: 6 pages + references (NeurIPS workshop format)

**Structure:**
1. Abstract (150 words)
2. Introduction (1 page) — Multi-agent trading is fragile; C-Gate provides robustness
3. Related Work (0.5 page) — FinVault, robust RL, LLM-agent safety
4. Method (1.5 pages) — Architecture, C-Gate Δ metric, three-regime policy, Guardian
5. Experiments (1.5 pages) — Setup, baselines, attack model, statistical tests
6. Results (1 page) — Cross-asset MaxDD reduction, adversarial robustness, ablations
7. Conclusion (0.5 page) — Limitations, future work

**Key framing decisions:**
- Lead with MaxDD reduction, NOT Sharpe
- "Controlled experimental environment" — single-asset isolation → multi-asset validation
- "Conservative cost assumptions (15 bps)" — strengthens credibility
- Acknowledge negative Sharpe explicitly: "The system prioritizes capital preservation
  over return maximization"
- Channel independence ablation is the architectural novelty claim

### 3.3 LaTeX formatting — ~1 day

Use NeurIPS 2026 workshop template. All figures 300 DPI, vector where possible.

---

## Week 4: Polish + Revisions

### 4.1 Internal review pass — ~1 day
- Check all claims have supporting evidence
- Verify all numbers in tables match experiment outputs
- Run spell/grammar check

### 4.2 Sensitivity analysis — ~1 day
- tau_low/tau_high percentile sensitivity (try p20/p70, p25/p75, p35/p85)
- Temperature sensitivity (try T=0.5, T=2.0 alongside T=1.0)
- Show results are robust to reasonable hyperparameter choices

### 4.3 Additional baselines if time permits — ~1 day
- Buy-and-Hold (already have)
- Simple momentum strategy (trend-following)
- Random policy
- These are cheap to implement and strengthen the comparison

### 4.4 Final revision — ~1 day

---

## Summary: What Changes and What Stays

### MUST DO (non-negotiable for workshop credibility)
| Item | Effort | Impact |
|------|--------|--------|
| Add 4 more tickers (MSFT, GOOGL, JPM, SPY) | 2 days | Critical |
| Fix transaction costs in eval scripts | 0.5 day | Critical |
| Statistical tests (bootstrap CI, paired tests) | 1 day | Critical |
| Fused-LLM-RL ablation baseline | 1 day | High |
| Paper draft with proper framing | 3 days | Critical |
| Publication-ready figures | 1 day | Critical |

### SHOULD DO (strengthens submission significantly)
| Item | Effort | Impact |
|------|--------|--------|
| Improve PPO (smaller network + noise augmentation) | 2 days | High |
| Semantic attacks | 1 day | Medium-High |
| Sensitivity analysis (tau, T) | 1 day | Medium |
| Additional simple baselines | 0.5 day | Medium |

### NICE TO HAVE (if time permits)
| Item | Effort | Impact |
|------|--------|--------|
| Coordinated attacks | 0.5 day | Low-Medium |
| Second test period | 1 day | Medium |
| Feature reduction experiments | 0.5 day | Low |
| Different RL algorithm (DQN) | 2 days | Medium |

### Total estimated effort: ~18 working days (fits in 3-4 weeks)
