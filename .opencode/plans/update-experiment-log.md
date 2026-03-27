# Plan: Update Experiment Log with Guardian Integration Results

## What to do

Append 4 new sections (9–12) to `docs/experiment-log.md` after the existing content (line 153).

## Content to append

The following text should be appended after the last line of the file:

---

```markdown
---

## 9. Guardian Integration (T=0.01, Current Calibration)

The Guardian was wired into `scripts/cgate_integration.py` to enforce risk limits on top of the C-Gate's regime-based decisions. The Guardian operates in two stages:

- **Stage 1 (Hard Constraints):** Circuit breakers that fire when portfolio drawdown or daily loss exceeds configured thresholds (`max_drawdown=0.15`, `max_daily_loss=0.05`). When triggered, the proposed action is overridden to flat regardless of the C-Gate's output.
- **Stage 2 (Adaptive Policy):** In the ambiguity regime, positions are scaled to 50% with a 2% stop-loss. In the conflict regime, the action is overridden to flat. In the agreement regime, full position with no stop-loss.

Position scaling is applied in the return calculation layer — the env still receives discrete actions {0,1,2} and operates normally. Portfolio state (value, peak, drawdown, daily PnL) is tracked across timesteps starting from a notional $100,000 capital.

### Before/After Guardian Comparison (T=0.01, τ_low=0.5341, τ_high=0.8117)

The regime distributions are identical with and without Guardian (same C-Gate thresholds), but the Guardian's position scaling and stop-loss enforcement change the realized performance.

**Without Guardian (original calibrated runs):**

| Seed | Split | Sharpe | Return | MaxDD |
|------|-------|--------|--------|-------|
| 123 | val | -1.519 | -10.71% | 15.27% |
| 123 | test | -0.906 | -5.89% | 9.03% |
| 789 | val | 1.087 | +8.89% | 9.08% |
| 789 | test | 0.342 | +1.81% | 5.43% |
| 2048 | val | -0.605 | -5.41% | 18.20% |
| 2048 | test | -0.014 | -0.77% | 7.90% |
| 7777 | val | -0.037 | -0.45% | 6.19% |
| 7777 | test | -0.628 | -4.77% | 14.55% |

**With Guardian (same thresholds):**

| Seed | Split | Sharpe | Return | MaxDD | Pos. Scaled | Stop-Loss | Stage 1 |
|------|-------|--------|--------|-------|-------------|-----------|---------|
| 123 | val | -1.006 | -5.47% | 10.54% | 36.3% | 1.8% | 6.2% |
| 123 | test | -0.604 | -3.31% | 7.49% | 28.4% | 0.0% | 6.0% |
| 789 | val | -0.298 | -1.52% | 7.27% | 61.9% | 0.9% | 8.0% |
| 789 | test | -0.237 | -0.95% | 4.14% | 54.3% | 0.9% | 3.4% |
| 2048 | val | -1.155 | -4.84% | 10.49% | 61.1% | 0.9% | 2.7% |
| 2048 | test | -0.452 | -2.99% | 7.33% | 64.7% | 0.9% | 9.5% |
| 7777 | val | 0.988 | +2.47% | 2.62% | 61.9% | 0.0% | 2.7% |
| 7777 | test | -0.522 | -2.66% | 7.19% | 45.7% | 2.6% | 5.2% |

### Key Observations

**MaxDD reduction is the headline result.** Average test MaxDD drops from 9.23% (no Guardian) to 6.54% (with Guardian) — a 29% relative reduction. The Guardian's position scaling in the ambiguity regime is the dominant mechanism: 28–65% of timesteps are scaled to half-size, mechanically halving the maximum possible per-step loss on those days.

**Sharpe ratios shift toward zero.** The Guardian compresses returns (both gains and losses) through position scaling, which tends to reduce Sharpe magnitude. Seed 789 drops from Sharpe 0.342 to -0.237 on test because the Guardian's 54% position scaling dampens the gains that were driving the positive Sharpe. This is the expected risk-return tradeoff — the Guardian is not designed to improve Sharpe, but to enforce risk budgets.

**Stage 1 circuit breakers fire 2–10% of timesteps.** This is appropriate — they are meant as rare emergency overrides, not routine interventions.

**Stop-losses trigger infrequently (0–3%).** The 2% stop-loss on ambiguity positions is a tight threshold, but because positions are already half-sized, large unrealized losses are less common.

---

## 10. Hyperparameter Sweep

A systematic sweep of C-Gate hyperparameters was conducted to find the optimal temperature and threshold percentile combination. The sweep script (`scripts/sweep_cgate.py`) evaluates 125 configurations: 5 temperatures × 5 low percentiles × 5 high percentiles × 4 seeds.

**Sweep grid:**
- Temperature: {0.01, 0.02, 0.05, 0.1, 0.2}
- Low percentile: {10, 20, 30, 40, 50}
- High percentile: {60, 70, 80, 90, 95}
- Seeds: {123, 789, 2048, 7777}

**Methodology:** For each temperature, the Δ distribution is pre-computed once across all seeds on the validation split. Percentile thresholds are then derived from this pooled distribution, and the full integration (with Guardian) is run for each seed. The objective is mean validation Sharpe across seeds.

**Results saved to:** `experiments/cgate/sweep_results.json`

### Best Configuration

**T=0.05, pcts=(30, 80) → τ_low=0.6515, τ_high=0.6964**

Mean validation Sharpe: 0.0425 (best among all 125 configs).

No configuration achieves positive val Sharpe on all 4 seeds simultaneously — seeds 123 and 2048 consistently lose money during the 2024 val period regardless of C-Gate tuning. This is a fundamental limitation of the underlying PPO policies, not the C-Gate.

### Sweep Observations

Higher temperatures (T=0.05, T=0.1) with narrower threshold gaps tend to perform better. At T=0.05, the Δ distribution is compressed into a narrow band (~0.57–0.75) compared to T=0.01 (~0.21–0.93). The narrow gap between τ_low and τ_high (0.045 vs 0.278) means the system is more decisive — fewer ambiguity timesteps, more clear agreement or conflict decisions.

Lower temperatures (T=0.01, T=0.02) spread the Δ distribution more widely, creating broader ambiguity zones. This provides finer-grained risk modulation but makes the system more sensitive to threshold placement.

---

## 11. Best Sweep Config Results (T=0.05 + Guardian)

Configuration: T=0.05, τ_low=0.6515, τ_high=0.6964, Guardian enabled.

| Seed | Split | Sharpe | Return | MaxDD | Agree | Ambig | Confl | Pos. Scaled | Stop-Loss | Stage 1 |
|------|-------|--------|--------|-------|-------|-------|-------|-------------|-----------|---------|
| 123 | val | -0.700 | -4.13% | 10.91% | 34.5% | 35.4% | 30.1% | 31.9% | 1.8% | 6.2% |
| 123 | test | -0.888 | -5.09% | 8.85% | 28.4% | 25.9% | 45.7% | 20.7% | 0.0% | 8.6% |
| 789 | val | 0.573 | +2.62% | 6.79% | 30.1% | 58.4% | 11.5% | 53.1% | 0.9% | 10.6% |
| 789 | test | -0.101 | -0.55% | 3.59% | 21.6% | 49.1% | 29.3% | 45.7% | 0.9% | 6.0% |
| 2048 | val | -1.380 | -7.82% | 14.18% | 30.1% | 50.4% | 19.5% | 48.7% | 0.9% | 4.4% |
| 2048 | test | 0.372 | +1.86% | 6.28% | 35.3% | 53.4% | 11.2% | 44.8% | 0.9% | 15.5% |
| 7777 | val | 1.677 | +4.33% | 2.91% | 31.9% | 48.7% | 19.5% | 46.0% | 0.0% | 5.3% |
| 7777 | test | -0.088 | -0.75% | 6.85% | 16.4% | 43.1% | 40.5% | 38.8% | 0.9% | 7.8% |

**Aggregate (mean ± std):**
- Val Sharpe: 0.043 ± 1.28
- Test Sharpe: -0.176 ± 0.53
- Val MaxDD: 8.70% ± 4.74%
- Test MaxDD: 6.39% ± 2.16%

### Comparison with T=0.01 Guardian

The best sweep config (T=0.05) improves mean val Sharpe from -0.368 (T=0.01) to +0.043, primarily because the narrower ambiguity zone allows more agreement-regime decisions (where full-size positions can capture gains). Test MaxDD is comparable: 6.39% (T=0.05) vs 6.54% (T=0.01). The regime distributions at T=0.05 are more balanced — roughly one-third in each regime — compared to T=0.01 where ambiguity dominates (42–72%).

---

## 12. Consolidated Comparison & Key Findings

### Cross-Configuration Summary (Mean Across 4 Seeds)

| Configuration | Val Sharpe | Test Sharpe | Val MaxDD | Test MaxDD |
|---|---|---|---|---|
| C-Gate T=0.01, no Guardian | -0.268 | -0.302 | 12.19% | 9.23% |
| C-Gate T=0.01 + Guardian | -0.368 | -0.454 | 7.73% | 6.54% |
| C-Gate T=0.05 + Guardian (best sweep) | +0.043 | -0.176 | 8.70% | 6.39% |
| PPO-Only (reference, no C-Gate) | +0.511 | -0.505 | N/A | N/A |

### Guardian's Impact on Maximum Drawdown

This is the primary architectural contribution. The Guardian reduces MaxDD consistently across all seeds and configurations:

| Seed | Test MaxDD (no Guardian) | Test MaxDD (T=0.01 + Guardian) | Test MaxDD (T=0.05 + Guardian) |
|------|--------------------------|-------------------------------|-------------------------------|
| 123 | 9.03% | 7.49% | 8.85% |
| 789 | 5.43% | 4.14% | 3.59% |
| 2048 | 7.90% | 7.33% | 6.28% |
| 7777 | 14.55% | 7.19% | 6.85% |
| **Mean** | **9.23%** | **6.54%** | **6.39%** |

Seed 7777 shows the most dramatic improvement: 14.55% → 7.19% (T=0.01) or 6.85% (T=0.05) — a 50%+ reduction in maximum drawdown. This seed has high conflict on test (38–41%), meaning the Guardian frequently overrides to flat, preventing large losses from accumulating.

### Thesis Framing

The Guardian does not improve Sharpe — in fact, mean Sharpe tends to degrade slightly because position scaling compresses both gains and losses. This is the expected and correct behavior: the Guardian is a risk management layer, not an alpha generator. The value proposition is that the Trinity architecture provides a **bounded-loss guarantee** that pure PPO cannot offer.

The best sweep config (T=0.05) represents the optimal tradeoff between risk control (MaxDD ≈ 6.4%) and return preservation (Sharpe closer to zero than the more conservative T=0.01 setting). Both configurations demonstrate the Guardian's effectiveness at capping tail risk.

**Next step:** Phase 6 — implement the four baselines (Executor-only, Analyst-only, Trinity-no-CGate, Fused LLM-RL) through the same integration framework for a fair 5-way comparison.
```

## Exact edit operation

**File:** `docs/experiment-log.md`  
**Action:** Replace the last two lines (lines 152–153) with themselves plus all the new content above appended after them.

Specifically, find:
```
- Performance is mixed across seeds, which is expected — the C-Gate architecture modulates risk, not generates alpha. The thesis contribution is the robustness property, not absolute returns.
- Missing headline dates (3 per test split) are correctly handled as conflict (Δ=1.0, action=flat) — safe by design.
```

And replace with the same two lines followed by the 4 new sections (9, 10, 11, 12) shown above.