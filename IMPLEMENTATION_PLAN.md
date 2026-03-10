# Robust Trinity — Implementation Plan

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Author:** Master's Thesis (One Semester, Part-Time)
**Timeline:** ~10-12 weeks, 3-4 hrs/day

---

## Architecture Summary

The Robust Trinity is a three-component multi-agent financial trading system coordinated by a Consistency Gate (C-Gate):

| Component | Type | Input Channel | Output |
|-----------|------|---------------|--------|
| **Analyst** | LLM (Claude / Llama) | Text (headlines, filings) | `(decision, reasoning)` |
| **Executor** | PPO (SB3) | Numeric (14 z-norm features × 30d lookback) | Policy distribution `π_RL` over {flat, long, short} |
| **Guardian** | Rule-based | All signals + portfolio state | Accept / reject / modify actions |
| **C-Gate** | Δ = 1 - π_RL(d_LLM) | `d_LLM` and `π_RL` | Regime: agreement / ambiguity / conflict |

**Core invariant:** Analyst and Executor share ZERO features (channel independence).

---

## Phase Summary

| Phase | Title | Weeks | Est. Hours | Tasks | Document |
|-------|-------|-------|------------|-------|----------|
| P0 | Environment & Project Scaffold | 1 | 8-12 | 4 | [docs/phase-0-setup.md](docs/phase-0-setup.md) |
| P1 | Data Pipeline | 1-2 | 10-15 | 3 | [docs/phase-1-data.md](docs/phase-1-data.md) |
| P2 | Executor Agent (PPO) | 2-3 | 20-30 | 5 | [docs/phase-2-executor.md](docs/phase-2-executor.md) |
| P3 | Analyst Agent (LLM) | 1-2 | 8-12 | 3 | [docs/phase-3-analyst.md](docs/phase-3-analyst.md) |
| P4 | Consistency Gate (C-Gate) | 1 | 6-8 | 2 | [docs/phase-4-cgate.md](docs/phase-4-cgate.md) |
| P5 | Guardian Agent | 1-2 | 10-15 | 3 | [docs/phase-5-guardian.md](docs/phase-5-guardian.md) |
| P6 | System Integration & Baselines | 1-2 | 12-18 | 3 | [docs/phase-6-integration.md](docs/phase-6-integration.md) |
| P7 | Adversarial Evaluation | 2-3 | 15-25 | 3 | [docs/phase-7-adversarial.md](docs/phase-7-adversarial.md) |
| P8 | Final Evaluation & Figures | 1-2 | 15-20 | 5 | [docs/phase-8-evaluation.md](docs/phase-8-evaluation.md) |
| **Total** | | **~10-12** | **~104-155** | **33** | |

---

## Task Index (33 Tasks)

### Phase 0: Environment & Project Scaffold
- **P0-T1:** Initialize project structure and Git repository
- **P0-T2:** Seed management and utility setup
- **P0-T3:** Fork and modernize ABIDES simulator
- **P0-T4:** Verify development toolchain

### Phase 1: Data Pipeline
- **P1-T1:** Download AAPL OHLCV data and engineer numeric features
- **P1-T2:** Collect and process text data (headlines/filings)
- **P1-T3:** Build data loaders and split management

### Phase 2: Executor Agent (PPO)
- **P2-T1:** Build custom TradingEnv (Gymnasium)
- **P2-T2:** Train PPO agent with differential Sharpe reward
- **P2-T3:** Policy extraction and distribution interface
- **P2-T4:** Hyperparameter tuning (W&B sweep)
- **P2-T5:** Executor evaluation and baseline metrics

### Phase 3: Analyst Agent (LLM)
- **P3-T1:** TradeSignal Pydantic schema and prompt engineering
- **P3-T2:** LLM backend (ollama dev, Claude prod)
- **P3-T3:** Precompute all Analyst signals and validation

### Phase 4: Consistency Gate (C-Gate)
- **P4-T1:** Divergence computation Δ = 1 - π_RL(d_LLM)
- **P4-T2:** ConsistencyGate class with three-regime policy

### Phase 5: Guardian Agent
- **P5-T1:** Hard constraint checks (Stage 1)
- **P5-T2:** Adaptive policy under ambiguity/conflict (Stage 2)
- **P5-T3:** Composite Guardian with edge case handling

### Phase 6: System Integration & Baselines
- **P6-T1:** TrinityPipeline end-to-end orchestrator
- **P6-T2:** Implement 4 baselines
- **P6-T3:** Backtester with metrics computation

### Phase 7: Adversarial Evaluation
- **P7-T1:** Numeric attacks (noise injection, feature perturbation)
- **P7-T2:** Semantic attacks (headline manipulation)
- **P7-T3:** Coordinated attacks and ABIDES adversarial agents

### Phase 8: Final Evaluation & Figures
- **P8-T1:** Experiment runner and configuration management
- **P8-T2:** Full evaluation suite execution
- **P8-T3:** C-Gate analysis (threshold sensitivity, regime distribution)
- **P8-T4:** Publication-quality tables and figures
- **P8-T5:** Statistical significance tests and final report

---

## Priority Ordering

| Priority | Scope | Rationale |
|----------|-------|-----------|
| **P0 (Must have)** | PPO agent + C-Gate + 1 attack type | Minimum viable thesis result |
| **P1 (Should have)** | LLM agent + Guardian + all 3 attack types | Full Trinity demonstration |
| **P2 (Nice to have)** | Threshold sensitivity + Fused LLM-RL baseline | Stronger ablation story |
| **P3 (Stretch)** | Nasdaq-100 backtest + publication-quality figures | Publishability |

---

## Dependency Graph

```
P0-T1 ──→ P0-T2 ──→ P0-T4
  │                    
  └──→ P0-T3          
                       
P0-T2 ──→ P1-T1 ──→ P1-T3
            │          │
P0-T4 ──→ P1-T2      │
                       │
P1-T3 ──→ P2-T1 ──→ P2-T2 ──→ P2-T3 ──→ P2-T4 ──→ P2-T5
                                  │
P1-T2 ──→ P3-T1 ──→ P3-T2 ──→ P3-T3
                                  │
P2-T3 + P3-T3 ──→ P4-T1 ──→ P4-T2
                                  │
P4-T2 ──→ P5-T1 ──→ P5-T2 ──→ P5-T3
                                  │
P5-T3 + P2-T5 ──→ P6-T1 ──→ P6-T2 ──→ P6-T3
                                          │
P6-T3 ──→ P7-T1 ──→ P7-T2 ──→ P7-T3
                                  │
P7-T3 ──→ P8-T1 ──→ P8-T2 ──→ P8-T3 ──→ P8-T4 ──→ P8-T5
```

---

## Key Technical Decisions (Locked)

| Decision | Choice |
|----------|--------|
| Hardware | MacBook M-series, no external GPU. Free Colab for sweeps only. |
| PPO Framework | Stable-Baselines3 (primary) + CleanRL (reference) |
| RL Environment | Custom Gymnasium env, obs_dim=423, Discrete(3) |
| PPO Architecture | 2×64 MLP, separate actor/critic, ~15-25K params |
| Reward | Differential Sharpe ratio + transaction cost penalty |
| LLM Dev | Local Llama 8B via ollama ($0), Claude Sonnet for final runs (~$25-50) |
| LLM Integration | Pre-compute all signals offline, lookup during simulation |
| ABIDES | Fork abides-jpmc-public, modernize gym→gymnasium, remove ray |
| Experiment Tracking | Weights & Biases (free academic tier) |
| Data | AAPL daily bars. Train: Jan 2023-Jun 2024, Test: Jul-Dec 2024 |
| Features | 14 z-normalized numeric features (log_return, rsi, macd_line, macd_signal, macd_histogram, bb_upper, bb_lower, bb_percent_b, atr, volume_ratio, realized_vol, close, high_low_range, close_open_return) |
| Action Space | Position-target: {flat=0, long=1, short=2} |
| C-Gate Metric | Δ = 1 - π_RL(d_LLM), thresholds τ_low=0.1, τ_high=0.4 |
| Budget | ~$25-80 API total, MacBook (owned), free Colab, free W&B |

---

## 4 Baselines

1. **Executor-only** — PPO agent with no LLM or Guardian
2. **Analyst-only** — LLM signals executed directly, no RL
3. **Trinity-no-CGate** — Simple agreement check: if argmax(π_RL)==d_LLM, full conviction; else execute argmax(π_RL) at 50% position size. No Δ threshold logic.
4. **Fused LLM-RL** — Intentionally breaks channel independence (ablation)
