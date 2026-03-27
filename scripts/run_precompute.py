"""Run Analyst pre-computation on all headlines.

Supports three backends:
  - gpt5 (default): Azure OpenAI GPT-5, uses AZURE_OPENAI_API_KEY.
  - ollama: Local Llama 3.1 8B, free, for development.
  - claude: Anthropic Claude Sonnet, requires ANTHROPIC_API_KEY.

Usage:
    python scripts/run_precompute.py                   # gpt5 → precomputed_signals_gpt5.json
    python scripts/run_precompute.py --backend ollama   # ollama → precomputed_signals.json
    python scripts/run_precompute.py --backend claude   # claude → precomputed_signals_claude.json
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.analyst.client import AnalystClient, AzureOpenAIBackend, ClaudeBackend, OllamaBackend
from src.analyst.precompute import precompute_signals

logging.basicConfig(level=logging.INFO)

HEADLINES_PATH = "data/processed/headlines.json"
OUTPUT_PATHS = {
    "gpt5": "data/processed/precomputed_signals_gpt5.json",
    "ollama": "data/processed/precomputed_signals.json",
    "claude": "data/processed/precomputed_signals_claude.json",
}


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Analyst signals")
    parser.add_argument(
        "--backend",
        choices=["gpt5", "ollama", "claude"],
        default="gpt5",
        help="LLM backend to use (default: gpt5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between API calls in seconds (default: 0.5 for ollama, 0.3 for claude)",
    )
    args = parser.parse_args()

    # Load headlines
    with open(HEADLINES_PATH, "r") as f:
        headlines = json.load(f)

    print(f"Loaded {len(headlines)} headlines from {HEADLINES_PATH}")

    # Select backend
    if args.backend == "gpt5":
        backend = AzureOpenAIBackend(model="gpt-5-chat", temperature=0.0)
        delay = args.delay if args.delay is not None else 0.3
        print("Using Azure OpenAI GPT-5 backend")
    elif args.backend == "claude":
        backend = ClaudeBackend(model="claude-sonnet-4-20250514", temperature=0.0)
        delay = args.delay if args.delay is not None else 0.3
        print("Using Claude Sonnet backend (requires ANTHROPIC_API_KEY)")
    else:
        backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
        delay = args.delay if args.delay is not None else 0.5
        print("Using Ollama (Llama 3.1 8B) backend")

    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)
    output_path = OUTPUT_PATHS[args.backend]

    # Run pre-computation
    cache = precompute_signals(
        headlines=headlines,
        client=client,
        output_path=output_path,
        delay=delay,
        save_every=10,
    )

    # Validate all signals
    from src.analyst.schema import TradeSignal

    errors = []
    for key, entry in cache.items():
        try:
            TradeSignal(
                reasoning=entry["reasoning"],
                decision=entry["decision"],
            )
        except Exception as e:
            errors.append((key, str(e)))

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for key, err in errors[:10]:
            print(f"  {key}: {err}")
    else:
        print(f"\nAll {len(cache)} signals validated successfully.")
        print(f"Output saved to {output_path}")

    # Print decision distribution
    decisions = [entry["decision"] for entry in cache.values()]
    from collections import Counter
    dist = Counter(decisions)
    total = len(decisions)
    print(f"\nDecision distribution ({args.backend}):")
    for d in ["hold", "buy", "sell"]:
        count = dist.get(d, 0)
        print(f"  {d:6s}: {count:4d} ({count/total*100:5.1f}%)")


if __name__ == "__main__":
    main()
