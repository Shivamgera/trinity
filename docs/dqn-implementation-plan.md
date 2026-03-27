# DQN Executor Implementation Plan

**Status:** In Progress
**Goal:** Build a DQN executor as a parallel alternative to PPO, then compare performance.
**Constraint:** NO modifications to any existing PPO files. DQN runs as a parallel pipeline.

---

## Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Seed selection metric | Q-Value Spread: `(max(Q) - mean(Q)) / std(Q)` | Measures decisiveness directly from Q-values; more natural for DQN than entropy-ratio |
| Hyperparameter tuning | W&B Bayesian sweep (~50 trials) before multiseed | More rigorous; avoids PPO-transplanted hyperparams being suboptimal for off-policy DQN |
| Timestep budget | 200k-300k max | Small dataset (756 train days); rely on early stopping (patience=10) to prevent memorization |
| Normalization | Static pre-normalization from training set only | No dynamic VecNormalize with off-policy replay buffer; compute mean/std on train split, apply frozen constants to all splits |
| Integration | None until DQN proves better than PPO | No C-Gate/adversarial integration until standalone performance is validated |

---

## Architecture Differences: DQN vs PPO

| Aspect | PPO (existing) | DQN (new) |
|---|---|---|
| Algorithm type | On-policy (actor-critic) | Off-policy (value-based) |
| Network | Separate pi=[64,64] and vf=[64,64] heads via `mlp_extractor` + `action_net` | Single Q-network with `net_arch=[64,64]`, outputs Q(s,a) for all 3 actions |
| Action selection | Sample from softmax(logits) | Epsilon-greedy over Q-values |
| Exploration | Entropy coefficient (ent_coef=0.001) | Epsilon schedule: 1.0 -> 0.05 over exploration_fraction of training |
| Experience storage | Rollout buffer (discarded after each update) | Replay buffer (persistent, sampled randomly) |
| Normalization | VecNormalize (dynamic) with norm_obs=False, norm_reward=True | Static pre-normalization; no VecNormalize at all |
| Probability extraction | `softmax(logits/T)` from `action_net(mlp_extractor(features))` | `softmax(Q-values/T)` from `q_net.forward()` |
| Callbacks | `_on_rollout_end` fires after each rollout collection | `_on_rollout_end` does NOT fire (off-policy); must use `_on_step` with frequency check |

---

## Files to Create

### 1. `src/executor/static_normalize.py`
Compute mean and std from training set features, save as JSON, provide a wrapper that applies frozen normalization to any environment.

- Load `aapl_features.parquet` for training split only
- Compute per-feature mean and std (shape: 143 = 10*14 + 3)
- Actually: compute mean/std for the 14 raw features, then the lookback flattening and portfolio state are handled at env level
- **Key insight:** The features in `aapl_features.parquet` are already rolling-z-normalized. The portfolio state (position, unrealized_pnl, time_since_trade) is already bounded [−1,1], [−1,1], [0,1]. So "static normalization" means computing mean/std of the **full 143-dim observation vectors** over training episodes, then applying `(obs - mean) / std` at inference.
- Save stats to `experiments/executor_dqn/normalization_stats.json`
- Provide `StaticNormalizer` class with `.normalize_obs(obs)` matching VecNormalize interface

### 2. `src/executor/policy_dqn.py`
DQN-specific policy extraction utilities, mirroring `policy.py` for PPO.

- `_extract_q_values(model, obs_tensor)` -> Q-values tensor (B, 3)
- `get_policy_distribution(model, obs, normalizer, temperature)` -> probs (3,)
- `get_policy_distribution_batch(model, obs_batch, normalizer, temperature)` -> probs (B, 3)
- `compute_q_value_spread(model, obs, normalizer)` -> float: `(max(Q) - mean(Q)) / std(Q)`
- `compute_q_value_spread_batch(model, obs_batch, normalizer)` -> (B,) array
- `load_executor_dqn(model_dir)` -> (DQN model, StaticNormalizer)

### 3. `src/executor/train_dqn.py`
DQN training script (standalone, no sweep).

- Uses `DQN` from stable_baselines3
- Single DummyVecEnv (n_envs=1, DQN requirement for replay buffer)
- **No VecNormalize** -- uses StaticNormalizer wrapper instead
- `StaticNormWrapper(gym.Wrapper)` that normalizes observations using pre-computed stats
- Early stopping via custom `DQNValCheckpointCallback` using `_on_step` with eval frequency
- W&B logging callback adapted for off-policy (no rollout end)

### 4. `src/executor/sweep_train_dqn.py`
W&B sweep training script for DQN hyperparameter search.

- Reads hyperparams from `wandb.config`
- Builds DQN with StaticNormWrapper
- Uses `DQNValCheckpointCallback` for early stopping
- Reports `val/sharpe_ratio` as sweep metric

### 5. `scripts/run_multiseed_dqn.py`
Multi-seed DQN training + seed selection.

- 20 seeds, same as PPO
- Uses best hyperparams from sweep
- Seed selection via **Q-Value Spread** diagnostic instead of entropy-ratio
- `select_and_freeze()` with combined score: `val_sharpe + 0.5 * test_sharpe`
- Outputs to `experiments/executor_dqn/multiseed/` and `experiments/executor_dqn/frozen/`

### 6. `configs/dqn_hyperparams.yaml`
Fixed DQN hyperparameters (baseline, pre-sweep).

### 7. `configs/dqn_sweep.yaml`
W&B Bayesian sweep config for DQN.

---

## W&B Sweep Parameters

### Fixed (not tuned)
- `total_timesteps`: 250,000 (within 200k-300k budget)
- `batch_size`: 64 (match PPO)
- `gamma`: tuned
- `net_arch`: [64, 64] (match PPO frozen architecture)
- `activation_fn`: Tanh (match PPO frozen)
- `reward_type`: log_return
- `target_update_interval`: tuned
- `tau`: 1.0 (hard target updates; standard for vanilla DQN)
- `patience`: 10

### Tuned in sweep
- `learning_rate`: log_uniform [1e-4, 1e-3]
- `gamma`: uniform [0.90, 0.99]
- `buffer_size`: categorical [10000, 25000, 50000]
- `learning_starts`: categorical [500, 1000, 2000]
- `exploration_fraction`: uniform [0.2, 0.6]
- `exploration_final_eps`: uniform [0.01, 0.10]
- `target_update_interval`: categorical [500, 1000, 2000]
- `train_freq`: categorical [1, 4, 8]
- `gradient_steps`: categorical [1, 4]

---

## Evaluation Plan (post-multiseed)

Compare DQN frozen seeds vs PPO frozen seeds on:
1. **Val Sharpe** (mean +/- std across seeds)
2. **Test Sharpe** (mean +/- std)
3. **Val/Test MaxDD**
4. **Seed consistency** (std of Sharpe across seeds)
5. **Q-Value Spread** distribution (DQN decisiveness diagnostic)
6. **AAPL Buy-and-Hold benchmark** (Sharpe=0.762, MaxDD=11.75%)

Decision criterion: DQN replaces PPO as executor if it achieves:
- Higher mean test Sharpe, OR
- Comparable Sharpe with significantly lower MaxDD, OR
- Significantly better seed consistency (lower std)

If DQN underperforms, we keep PPO and document DQN as an ablation.

---

## Execution Sequence

### Day 1: Infrastructure
1. [x] Create implementation plan
2. [ ] `src/executor/static_normalize.py` -- static normalizer
3. [ ] `src/executor/policy_dqn.py` -- Q-value extraction
4. [ ] `src/executor/train_dqn.py` -- base training script
5. [ ] `configs/dqn_hyperparams.yaml` -- baseline config
6. [ ] Smoke test: train 1 seed for 10k steps, verify env works

### Day 2: Sweep
7. [ ] `src/executor/sweep_train_dqn.py` -- sweep training
8. [ ] `configs/dqn_sweep.yaml` -- sweep config
9. [ ] Launch W&B sweep (~50 trials)
10. [ ] Analyze sweep results, lock best hyperparams

### Day 3: Multiseed
11. [ ] `scripts/run_multiseed_dqn.py` -- multiseed with Q-value spread
12. [ ] Run 20-seed training with best hyperparams
13. [ ] Evaluate all seeds on val + test
14. [ ] Select and freeze top 4 seeds

### Day 4: Comparison
15. [ ] Head-to-head DQN vs PPO comparison
16. [ ] Decision: replace PPO or keep as ablation
17. [ ] If replacing: integrate with C-Gate/adversarial pipeline

---

## Static Normalization Design

### Problem
VecNormalize maintains a running mean/std that updates during training. With DQN's replay buffer, old experiences are re-sampled with stale normalization statistics, creating a distribution shift that destabilizes training.

### Solution
1. Collect observations from N full episodes of random-policy rollouts on the **training set only**
2. Compute `obs_mean` (shape 143,) and `obs_std` (shape 143,) with `ddof=0`
3. Save to JSON: `experiments/executor_dqn/normalization_stats.json`
4. Create `StaticNormWrapper(gym.Wrapper)` that applies `(obs - mean) / (std + 1e-8)` on every `reset()` and `step()` return
5. Use this wrapper for all DQN training, evaluation, and C-Gate integration
6. The same stats file is loaded at inference time -- no VecNormalize pkl needed

### Note on feature pre-normalization
The 14 numeric features in `aapl_features.parquet` are already rolling-z-normalized (252-day window). The 3 portfolio state features are already bounded. So the static normalization is a second-pass normalization of the full 143-dim observation to have zero mean and unit variance -- this mainly helps with the lookback-window flattening creating different scales across time lags.
