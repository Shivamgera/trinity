"""Analyst validation: accuracy vs FinancialPhraseBank, consistency, and Δ preview."""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Part 1: Accuracy vs FinancialPhraseBank
# ------------------------------------------------------------------


def load_phrasebank(path: str = "data/raw/", max_samples: int = 50) -> list[dict]:
    """Load FinancialPhraseBank sentences with labels.

    Supports two formats:
    - CSV with columns: sentence,label (from download_phrasebank.py)
    - Raw text with lines like: "sentence text"@label
    """
    import csv

    samples: list[dict] = []
    fpb_path = Path(path)

    # Try CSV first (from download_phrasebank.py)
    csv_files = list(fpb_path.glob("*phrasebank*.csv")) + list(
        fpb_path.glob("*FinancialPhraseBank*.csv")
    )
    # Then try raw text files
    txt_files = list(fpb_path.glob("*Sentences*")) + list(
        fpb_path.glob("*FinancialPhraseBank*.txt")
    )

    if csv_files:
        fpb_file = csv_files[0]
        logger.info(f"Loading FinancialPhraseBank CSV from {fpb_file}")
        with open(fpb_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip().lower()
                if label in ("positive", "negative", "neutral"):
                    samples.append({"sentence": row["sentence"], "label": label})
    elif txt_files:
        fpb_file = txt_files[0]
        logger.info(f"Loading FinancialPhraseBank TXT from {fpb_file}")
        with open(fpb_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "@" in line:
                    parts = line.rsplit("@", 1)
                    if len(parts) == 2:
                        sentence, label = (
                            parts[0].strip(),
                            parts[1].strip().lower(),
                        )
                        if label in ("positive", "negative", "neutral"):
                            samples.append({"sentence": sentence, "label": label})
    else:
        logger.warning(
            f"No FinancialPhraseBank files found in {path}. Skipping accuracy test."
        )
        return []

    # Sample evenly across labels
    by_label: dict[str, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    selected: list[dict] = []
    per_label = max_samples // 3
    for label, items in by_label.items():
        selected.extend(items[:per_label])

    logger.info(
        f"Selected {len(selected)} samples: "
        f"{Counter(s['label'] for s in selected)}"
    )
    return selected


def map_label_to_decision(label: str) -> str:
    """Map FinancialPhraseBank labels to our decision space."""
    return {"positive": "buy", "negative": "sell", "neutral": "hold"}[label]


def run_accuracy_test():
    """Test Analyst accuracy against FinancialPhraseBank ground truth."""
    from src.analyst.client import AnalystClient, OllamaBackend

    samples = load_phrasebank()
    if not samples:
        return None, None

    backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)

    correct = 0
    results = []

    for sample in samples:
        signal = client.analyze(sample["sentence"], "TEST", "2024-01-01")
        expected = map_label_to_decision(sample["label"])
        is_correct = signal.decision == expected
        correct += int(is_correct)
        results.append(
            {
                "sentence": sample["sentence"][:80],
                "label": sample["label"],
                "expected": expected,
                "predicted": signal.decision,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(results) if results else 0
    logger.info(f"Accuracy: {correct}/{len(results)} = {accuracy:.2%}")

    # Log to W&B if available
    try:
        import wandb

        wandb.init(project="robust-trinity", name="analyst-validation", reinit=True)
        wandb.log({"analyst/accuracy": accuracy, "analyst/n_samples": len(results)})
        wandb.log(
            {
                "analyst/results_table": wandb.Table(
                    columns=list(results[0].keys()),
                    data=[list(r.values()) for r in results],
                )
            }
        )
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")

    return results, accuracy


# ------------------------------------------------------------------
# Part 2: Inter-run Consistency (Cohen's kappa)
# ------------------------------------------------------------------


def run_consistency_test(n_runs: int = 3):
    """Run same headlines multiple times, measure agreement."""
    from src.analyst.client import AnalystClient, OllamaBackend

    samples = load_phrasebank(max_samples=50)
    if not samples:
        return None

    backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)

    # Collect decisions across runs
    all_decisions: dict[int, list[str]] = {i: [] for i in range(n_runs)}
    for run_idx in range(n_runs):
        logger.info(f"Consistency run {run_idx + 1}/{n_runs}")
        for sample in samples:
            signal = client.analyze(sample["sentence"], "TEST", "2024-01-01")
            all_decisions[run_idx].append(signal.decision)

    # Compute pairwise Cohen's kappa
    from sklearn.metrics import cohen_kappa_score

    kappas = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            kappa = cohen_kappa_score(all_decisions[i], all_decisions[j])
            kappas.append(kappa)
            logger.info(f"Kappa (run {i} vs {j}): {kappa:.3f}")

    mean_kappa = float(np.mean(kappas))
    logger.info(f"Mean Cohen's kappa: {mean_kappa:.3f}")

    try:
        import wandb

        wandb.log({"analyst/mean_kappa": mean_kappa})
    except Exception:
        pass

    return mean_kappa


# ------------------------------------------------------------------
# Part 3: Δ Preview — Analyst vs Executor Divergence
# ------------------------------------------------------------------
# Action mapping: must match Executor's action space
DECISION_TO_INDEX = {"hold": 0, "buy": 1, "sell": 2}


def run_delta_preview():
    """Compare Analyst decisions against Executor policy distributions
    on overlapping dates. Computes Δ = 1 - π_RL(d_LLM | s_t).
    This previews C-Gate behavior.

    Uses the TradingEnv to generate proper 143-dim observations (with
    lookback window + portfolio state) and matches by date.
    """
    from src.analyst.schema import TradeSignal

    # Load precomputed signals — build date->signal lookup
    signals_path = "data/processed/precomputed_signals.json"
    if not Path(signals_path).exists():
        logger.error(
            f"Precomputed signals not found at {signals_path}. Run P3-T2 first."
        )
        return None

    with open(signals_path, "r") as f:
        signals_cache = json.load(f)

    # Build date -> decision mapping from signals
    date_to_decision: dict[str, str] = {}
    for entry in signals_cache.values():
        date_to_decision[entry["date"]] = entry["decision"]

    logger.info(f"Loaded {len(date_to_decision)} signal dates.")

    # Load executor model + VecNormalize
    try:
        from src.executor.env_factory import make_trading_env
        from src.executor.policy import get_policy_distribution, load_executor

        model, vec_normalize = load_executor(
            "experiments/executor/frozen/seed_7777"
        )
    except Exception as e:
        logger.error(f"Could not load Executor model: {e}")
        return None

    # Run through val split to get observations and dates
    deltas = []
    dates = []

    for split in ["val", "test"]:
        try:
            env_fn = make_trading_env(
                split=split, random_start=False, episode_length=None
            )
            env = env_fn()
        except Exception as e:
            logger.warning(f"Could not create env for split={split}: {e}")
            continue

        obs, info = env.reset()
        done = False
        while not done:
            date = info.get("date", "")
            if date in date_to_decision:
                decision = date_to_decision[date]
                action_index = DECISION_TO_INDEX[decision]
                pi_rl = get_policy_distribution(model, obs, vec_normalize)
                delta = 1.0 - pi_rl[action_index]
                deltas.append(float(delta))
                dates.append(date)

            # Step with the model's preferred action (doesn't matter —
            # we only care about the observation at each step)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        env.close()
        logger.info(f"Split {split}: collected {len(deltas)} Δ values so far.")

    if not deltas:
        logger.warning("No overlapping dates found between signals and features.")
        return None

    deltas_arr = np.array(deltas)
    logger.info(f"Δ statistics over {len(deltas_arr)} dates:")
    logger.info(f"  Mean:   {deltas_arr.mean():.4f}")
    logger.info(f"  Median: {np.median(deltas_arr):.4f}")
    logger.info(f"  Std:    {deltas_arr.std():.4f}")
    logger.info(f"  Min:    {deltas_arr.min():.4f}")
    logger.info(f"  Max:    {deltas_arr.max():.4f}")

    # Regime counts with initial thresholds
    tau_low, tau_high = 0.1, 0.4
    agreement = float(np.sum(deltas_arr <= tau_low) / len(deltas_arr))
    ambiguity = float(
        np.sum((deltas_arr > tau_low) & (deltas_arr <= tau_high)) / len(deltas_arr)
    )
    conflict = float(np.sum(deltas_arr > tau_high) / len(deltas_arr))

    logger.info(f"  Agreement (Δ ≤ {tau_low}): {agreement:.1%}")
    logger.info(f"  Ambiguity ({tau_low} < Δ ≤ {tau_high}): {ambiguity:.1%}")
    logger.info(f"  Conflict  (Δ > {tau_high}): {conflict:.1%}")

    # Log to W&B
    try:
        import wandb

        wandb.init(project="robust-trinity", name="delta-preview", reinit=True)
        wandb.log(
            {
                "delta/mean": float(deltas_arr.mean()),
                "delta/median": float(np.median(deltas_arr)),
                "delta/std": float(deltas_arr.std()),
                "delta/pct_agreement": agreement,
                "delta/pct_ambiguity": ambiguity,
                "delta/pct_conflict": conflict,
            }
        )
        wandb.log({"delta/histogram": wandb.Histogram(deltas_arr, num_bins=50)})
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")

    return deltas_arr


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Accuracy vs FinancialPhraseBank")
    print("=" * 60)
    run_accuracy_test()

    print("\n" + "=" * 60)
    print("PART 2: Inter-run Consistency")
    print("=" * 60)
    run_consistency_test()

    print("\n" + "=" * 60)
    print("PART 3: Δ Preview (Analyst vs Executor Divergence)")
    print("=" * 60)
    run_delta_preview()
