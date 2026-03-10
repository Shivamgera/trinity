# Verification Scripts

These scripts verify the development toolchain. Run them in order:

1. `python scripts/verify_ollama.py` — Tests local Llama model via ollama
2. `python scripts/verify_sb3.py` — Tests PPO training via Stable-Baselines3
3. `python scripts/verify_wandb.py` — Tests experiment logging via W&B
