"""Batch pre-computation pipeline for Analyst signals."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from src.analyst.client import AnalystClient
from src.analyst.schema import TradeSignal

logger = logging.getLogger(__name__)


def headline_hash(headline: str, ticker: str, date: str) -> str:
    """Deterministic hash for a headline to use as cache key."""
    key = f"{ticker}|{date}|{headline}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_existing_cache(output_path: str) -> dict:
    """Load existing precomputed signals from disk."""
    path = Path(output_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, output_path: str) -> None:
    """Save cache to disk atomically (write to tmp, then rename)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
    tmp_path.rename(path)


def precompute_signals(
    headlines: list[dict],
    client: AnalystClient,
    output_path: str,
    delay: float = 0.5,
    save_every: int = 10,
) -> dict:
    """Pre-compute Analyst signals for all headlines.

    Each headline dict must have keys: "headline", "ticker", "date".

    Supports resumption: loads existing cache from output_path and skips
    already-computed headlines.

    Args:
        headlines: List of dicts with "headline", "ticker", "date" keys.
        client: Configured AnalystClient instance.
        output_path: Path to save/load the cache JSON.
        delay: Seconds to wait between LLM calls (rate limiting).
        save_every: Save cache to disk every N new computations.

    Returns:
        Dict mapping headline_hash -> serialized TradeSignal dict.
    """
    cache = load_existing_cache(output_path)
    new_count = 0

    logger.info(f"Loaded {len(cache)} existing signals from cache.")
    logger.info(
        f"Processing {len(headlines)} headlines "
        f"({len(headlines) - len(cache)} new)."
    )

    for item in tqdm(headlines, desc="Pre-computing Analyst signals"):
        h = item["headline"]
        ticker = item["ticker"]
        date = item["date"]
        key = headline_hash(h, ticker, date)

        if key in cache:
            continue  # already computed

        signal = client.analyze(h, ticker, date)
        cache[key] = {
            "headline": h,
            "ticker": ticker,
            "date": date,
            "reasoning": signal.reasoning,
            "decision": signal.decision,
        }
        new_count += 1

        if new_count % save_every == 0:
            save_cache(cache, output_path)
            logger.info(f"Checkpoint: saved {len(cache)} signals.")

        if delay > 0:
            time.sleep(delay)

    # Final save
    save_cache(cache, output_path)
    logger.info(f"Done. {new_count} new signals computed. Total: {len(cache)}.")
    return cache
