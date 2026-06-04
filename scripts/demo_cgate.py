# -*- coding: utf-8 -*-
"""Live demo script for thesis defence - C-Gate in action.

Run: python scripts/demo_cgate.py
"""

import numpy as np

from src.cgate.gate import ConsistencyGate

# Calibrated thresholds (p20/p80 from validation set)
gate = ConsistencyGate(tau_low=0.6329, tau_high=0.6991)

print("=" * 60)
print("C-GATE LIVE DEMO")
print("=" * 60)
print(f"\nThresholds: tau_low = {gate.tau_low}, tau_high = {gate.tau_high}")
print(f"Actions: 0 = flat, 1 = long, 2 = short\n")

# --- DEMO 1: Agreement ---
print("-" * 60)
print("SCENARIO 1: Normal operation (no attack)")
print("-" * 60)
pi_rl = np.array([0.15, 0.70, 0.15])
d_llm = "buy"
print(f"  Executor policy pi_RL = {pi_rl}  (70% long)")
print(f"  Analyst decision     = '{d_llm}'")
result = gate.evaluate(d_llm, pi_rl)
print(f"\n  Delta = 1 - pi_RL(long) = 1 - 0.70 = {result.delta:.2f}")
print(f"  Regime: {result.regime.upper()}")
print(f"  Action: {result.action} (long)")
print(f"  → System executes full position.\n")

input("  [Press Enter for next scenario...]")

# --- DEMO 2: Conflict (analyst poisoning) ---
print("\n" + "-" * 60)
print("SCENARIO 2: Analyst poisoned - signal flipped to 'sell'")
print("-" * 60)
d_llm_attack = "sell"
print(f"  Executor policy pi_RL = {pi_rl}  (unchanged - numeric channel clean)")
print(f"  Analyst decision     = '{d_llm_attack}' <- POISONED")
result = gate.evaluate(d_llm_attack, pi_rl)
print(f"\n  Delta = 1 - pi_RL(short) = 1 - 0.15 = {result.delta:.2f}")
print(f"  Regime: {result.regime.upper()}")
print(f"  Action: {result.action} (flat = FAIL-SAFE)")
print(f"  → Attack detected! System refuses to act on poisoned signal.\n")

input("  [Press Enter for next scenario...]")

# --- DEMO 3: Ambiguity (borderline) ---
print("\n" + "-" * 60)
print("SCENARIO 3: Executor uncertain - ambiguity regime")
print("-" * 60)
pi_rl_ambig = np.array([0.30, 0.35, 0.35])
d_llm_ambig = "buy"
print(f"  Executor policy pi_RL = {pi_rl_ambig}  (nearly uniform)")
print(f"  Analyst decision     = '{d_llm_ambig}'")
result = gate.evaluate(d_llm_ambig, pi_rl_ambig)
print(f"\n  Delta = 1 - pi_RL(long) = 1 - 0.35 = {result.delta:.2f}")
print(f"  Regime: {result.regime.upper()}")
print(f"  Action: {result.action} (long, but half-sized + 2% stop-loss)")
print(f"  → Uncertain - system hedges with conservative position.\n")

print("=" * 60)
print("KEY TAKEAWAY: No extra model trained. Divergence between two")
print("independent channels IS the detection signal.")
print("=" * 60)
