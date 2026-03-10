# Phase 3: Analyst Agent — LLM Integration

**Project:** Structural Fault Tolerance in Heterogeneous Financial AI Systems under Adversarial Signal Injection
**Project Root:** /Users/shivamgera/projects/research1
**Timeline:** Weeks 5–6 (~12–16 hours total across 3 tasks)

---

## P3-T1: Implement TradeSignal Schema and LLM Client

**Estimated time:** ~2.5 hours
**Dependencies:** None (Phase 0–2 assumed complete)

### Context

The Robust Trinity architecture requires an Analyst agent that processes financial headlines via an LLM and outputs structured trade signals. The Analyst is one of two independent channels — it processes ONLY text data, while the Executor (already built) processes ONLY numeric market data. Channel independence is a hard design constraint.

The Analyst output is a pair `(d, r)` where:
- `d` ∈ {hold, buy, sell} — the directional decision
- `r` — reasoning trace (chain-of-thought text)

The Analyst's decision d_LLM is compared against the Executor's policy distribution π_RL via the C-Gate divergence Δ = 1 - π_RL(d_LLM | s_t).

The project uses Python 3.11+, Pydantic v2 for schema enforcement, and supports two LLM backends: local ollama (Llama 8B) for development and Anthropic Claude for production.

### Objective

Build the Pydantic schema for trade signals, a dual-backend LLM client, and the prompt templates.

### Detailed Instructions

#### Step 1: Create `src/analyst/__init__.py`

Create an empty `__init__.py` to make `src/analyst` a Python package. If one already exists, leave it as-is.

#### Step 2: Create `src/analyst/schema.py`

```python
"""Trade signal schema with chain-of-thought enforcement."""
from pydantic import BaseModel, field_validator
from typing import Literal


class TradeSignal(BaseModel):
    """
    Analyst agent output schema.

    IMPORTANT: 'reasoning' is the FIRST field to force chain-of-thought
    before the model commits to a decision. This is a deliberate prompt
    engineering choice — LLMs that reason before deciding produce better-
    calibrated outputs.
    """
    reasoning: str
    decision: Literal["hold", "buy", "sell"]

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Reasoning must not be empty")
        return v
```

Key decisions:
- `reasoning` is the first field. This is intentional — when the LLM fills fields in order, it is forced to reason before committing to a decision.

#### Step 3: Create `src/analyst/prompts.py`

```python
"""System prompt and few-shot examples for the Analyst agent."""

SYSTEM_PROMPT = """You are a quantitative financial analyst specializing in short-term equity signals. Your task is to analyze a financial news headline and produce a structured trade signal.

You MUST respond with valid JSON matching this exact schema:
{
  "reasoning": "<your step-by-step analysis>",
  "decision": "<hold | buy | sell>"
}

RULES:
1. You MUST write your reasoning BEFORE choosing a decision. Think step by step.
2. Consider: Is this headline material? Is the direction clear? Are there caveats?
3. When in doubt, prefer "hold" over a forced directional call.
4. Do NOT use any numeric market data — you only analyze text.
5. Your response must be valid JSON and nothing else."""

FEW_SHOT_EXAMPLES = [
    {
        "headline": "Apple reports Q3 earnings above analyst expectations, revenue up 12% YoY",
        "ticker": "AAPL",
        "response": {
            "reasoning": "Earnings beat with 12% YoY revenue growth is a clear positive catalyst. Earnings surprises tend to drive short-term price appreciation. The signal is strong and directionally clear.",
            "decision": "buy"
        }
    },
    {
        "headline": "FDA rejects Pfizer's application for new cancer drug, citing insufficient trial data",
        "ticker": "PFE",
        "response": {
            "reasoning": "FDA rejection is a significant negative catalyst for a pharmaceutical company. This typically leads to immediate selling pressure. The signal is strong and clearly bearish.",
            "decision": "sell"
        }
    },
    {
        "headline": "Tesla announces new partnership with regional charging network provider",
        "ticker": "TSLA",
        "response": {
            "reasoning": "A regional charging partnership is mildly positive but not material to Tesla's overall business. This is incremental news, not a major catalyst. The market impact is likely minimal.",
            "decision": "hold"
        }
    },
    {
        "headline": "Microsoft faces antitrust investigation in EU over cloud bundling practices",
        "ticker": "MSFT",
        "response": {
            "reasoning": "Antitrust investigations are negative but outcomes are uncertain and typically take years. Short-term impact depends on market sentiment. This is a moderate negative signal with significant uncertainty.",
            "decision": "sell"
        }
    },
    {
        "headline": "Amazon Web Services experiences brief outage in US-East region",
        "ticker": "AMZN",
        "response": {
            "reasoning": "Brief outages happen regularly and are typically resolved quickly. Unless this is prolonged or causes major customer losses, the market impact is negligible. No clear directional signal.",
            "decision": "hold"
        }
    }
]


def format_user_prompt(headline: str, ticker: str, date: str) -> str:
    """Format the user prompt for a single headline analysis."""
    return f"Analyze this headline for {ticker} on {date}:\n\n\"{headline}\""


def format_few_shot_messages() -> list[dict]:
    """Format few-shot examples as a conversation history for chat APIs."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": format_user_prompt(ex["headline"], ex["ticker"], "2024-01-01")
        })
        import json
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["response"])
        })
    return messages
```

#### Step 4: Create `src/analyst/client.py`

```python
"""Dual-backend LLM client for the Analyst agent."""
import json
import time
import logging
from abc import ABC, abstractmethod

from src.analyst.schema import TradeSignal
from src.analyst.prompts import SYSTEM_PROMPT, format_few_shot_messages, format_user_prompt

logger = logging.getLogger(__name__)

# Neutral fallback signal used when all retries are exhausted
FALLBACK_SIGNAL = TradeSignal(
    reasoning="API error — returning neutral fallback signal.",
    decision="hold",
)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def call(self, system_prompt: str, messages: list[dict]) -> str:
        """Send messages to the LLM and return the raw string response."""
        ...


class OllamaBackend(LLMBackend):
    """Local Llama 8B via ollama Python library."""

    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        import ollama

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        response = ollama.chat(
            model=self.model,
            messages=full_messages,
            options={"temperature": self.temperature},
            format="json",  # ollama JSON mode
        )
        return response["message"]["content"]


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API with prompt caching."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0):
        import anthropic
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.model = model
        self.temperature = temperature

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # prompt caching
                }
            ],
            messages=messages,
        )
        return response.content[0].text


class AnalystClient:
    """
    Main Analyst client. Wraps an LLM backend and returns validated TradeSignal objects.

    Usage:
        client = AnalystClient(backend=OllamaBackend())
        signal = client.analyze("AAPL beats earnings", "AAPL", "2024-01-15")
    """

    def __init__(self, backend: LLMBackend, max_retries: int = 3, include_few_shot: bool = True):
        self.backend = backend
        self.max_retries = max_retries
        self.include_few_shot = include_few_shot

    def analyze(self, headline: str, ticker: str, date: str) -> TradeSignal:
        """
        Analyze a single headline and return a validated TradeSignal.

        Retries up to max_retries times on API or parsing failures.
        Returns FALLBACK_SIGNAL (hold) after all retries exhausted.
        """
        messages = []
        if self.include_few_shot:
            messages.extend(format_few_shot_messages())
        messages.append({"role": "user", "content": format_user_prompt(headline, ticker, date)})

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self.backend.call(SYSTEM_PROMPT, messages)
                parsed = json.loads(raw_response)
                signal = TradeSignal(**parsed)
                return signal
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries}: JSON parse error: {e}")
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries}: Error: {e}")

            if attempt < self.max_retries:
                time.sleep(1.0 * attempt)  # linear backoff

        logger.error(f"All {self.max_retries} retries exhausted for headline: {headline[:80]}...")
        return FALLBACK_SIGNAL
```

#### Step 5: Create tests

Create `tests/test_analyst_schema.py`:

```python
"""Tests for TradeSignal schema and AnalystClient."""
import pytest
import json
from unittest.mock import MagicMock, patch
from src.analyst.schema import TradeSignal
from src.analyst.client import AnalystClient, OllamaBackend, FALLBACK_SIGNAL, LLMBackend


class TestTradeSignal:
    def test_valid_signal(self):
        s = TradeSignal(reasoning="test reasoning", decision="buy")
        assert s.decision == "buy"

    def test_invalid_decision(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="test", decision="panic")

    def test_empty_reasoning_rejected(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="", decision="hold")

    def test_missing_fields(self):
        with pytest.raises(Exception):
            TradeSignal(decision="buy")  # missing reasoning


class MockFailingBackend(LLMBackend):
    def call(self, system_prompt, messages):
        raise ConnectionError("API down")


class MockValidBackend(LLMBackend):
    def call(self, system_prompt, messages):
        return json.dumps({
            "reasoning": "Looks bullish",
            "decision": "buy"
        })


class TestAnalystClient:
    def test_successful_analysis(self):
        client = AnalystClient(backend=MockValidBackend(), include_few_shot=False)
        signal = client.analyze("AAPL beats earnings", "AAPL", "2024-01-15")
        assert signal.decision == "buy"

    def test_fallback_on_failure(self):
        client = AnalystClient(backend=MockFailingBackend(), max_retries=2, include_few_shot=False)
        signal = client.analyze("test headline", "AAPL", "2024-01-15")
        assert signal.decision == "hold"
```

### Acceptance Criteria

- [ ] `TradeSignal` schema enforces field ordering (reasoning first) and valid decisions
- [ ] `AnalystClient.analyze()` returns a valid `TradeSignal` on success
- [ ] `AnalystClient.analyze()` returns `FALLBACK_SIGNAL` (hold, "API error") after exhausting retries
- [ ] `OllamaBackend` is callable with ollama JSON mode
- [ ] `ClaudeBackend` uses prompt caching via `cache_control`
- [ ] All tests in `tests/test_analyst_schema.py` pass: `pytest tests/test_analyst_schema.py -v`

### Files to Create/Modify

- `src/analyst/__init__.py` (create if not exists)
- `src/analyst/schema.py` (create)
- `src/analyst/prompts.py` (create)
- `src/analyst/client.py` (create)
- `tests/test_analyst_schema.py` (create)

### Dependencies

- Phase 0–2 assumed complete (Python env, ollama working, project structure exists)

### Human Checkpoint

- Manually run `AnalystClient(backend=OllamaBackend()).analyze("Apple reports record earnings", "AAPL", "2024-06-15")` and verify the response is a sensible `TradeSignal` with buy/hold decision and reasonable reasoning. If ollama returns unparseable JSON, check the ollama model version and `format="json"` support.

---

## P3-T2: Pre-computation Pipeline

**Estimated time:** ~2.5 hours
**Dependencies:** P3-T1 must be complete (schema.py, client.py, prompts.py exist and work)

### Context

The Robust Trinity architecture processes text and numeric data through independent channels. The Analyst produces structured signals `(d, r)`, and the Consistency Gate (C-Gate) computes the divergence Δ = 1 - π_RL(d_LLM | s_t) by querying the Executor's policy distribution π_RL at the action index corresponding to the Analyst's decision. No pseudo-distribution conversion is needed.

Additionally, since LLM calls are expensive and slow, we pre-compute all Analyst signals offline and cache them. During backtesting and C-Gate evaluation, we look up pre-computed signals by headline hash.

**Already available from previous tasks:**
- `src/analyst/schema.py` — `TradeSignal` Pydantic model
- `src/analyst/client.py` — `AnalystClient` with `OllamaBackend` and `ClaudeBackend`
- `data/processed/headlines.json` — headlines data from Phase 1
- `src/executor/policy.py` — `get_policy_distribution(model, obs) → np.array` from Phase 2

### Objective

Build the batch pre-computation pipeline that processes all headlines through the LLM and caches results.

### Detailed Instructions

#### Step 1: Note on C-Gate integration

**Note:** The Analyst outputs only (decision, reasoning). No pseudo-distribution conversion is needed — the C-Gate computes Δ = 1 - π_RL(d_LLM) directly using the Executor's policy distribution. The action mapping is: "hold"→flat(0), "buy"→long(1), "sell"→short(2).

#### Step 2: Create `src/analyst/precompute.py`

```python
"""Batch pre-computation pipeline for Analyst signals."""
import json
import hashlib
import logging
import time
from pathlib import Path
from tqdm import tqdm

from src.analyst.schema import TradeSignal
from src.analyst.client import AnalystClient

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
    """
    Pre-compute Analyst signals for all headlines.

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
        Dict mapping headline_hash → serialized TradeSignal dict.
    """
    cache = load_existing_cache(output_path)
    new_count = 0

    logger.info(f"Loaded {len(cache)} existing signals from cache.")
    logger.info(f"Processing {len(headlines)} headlines ({len(headlines) - len(cache)} new).")

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
```

#### Step 3: Create `scripts/run_precompute.py`

This is the entry-point script to run the full pre-computation on all headlines:

```python
"""Run Analyst pre-computation on all headlines using ollama."""
import json
import logging
from pathlib import Path

from src.analyst.client import AnalystClient, OllamaBackend
from src.analyst.precompute import precompute_signals

logging.basicConfig(level=logging.INFO)

HEADLINES_PATH = "data/processed/headlines.json"
OUTPUT_PATH = "data/processed/precomputed_signals.json"


def main():
    # Load headlines
    with open(HEADLINES_PATH, "r") as f:
        headlines = json.load(f)

    print(f"Loaded {len(headlines)} headlines from {HEADLINES_PATH}")

    # Use ollama for dev (free, local)
    backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)

    # Run pre-computation
    cache = precompute_signals(
        headlines=headlines,
        client=client,
        output_path=OUTPUT_PATH,
        delay=0.5,
        save_every=10,
    )

    # Validate
    from src.analyst.schema import TradeSignal

    errors = []
    for key, entry in cache.items():
        try:
            signal = TradeSignal(
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
        print(f"Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

#### Step 4: Write tests for conversion

Add to `tests/test_analyst_conversion.py`:

```python
"""Tests for Analyst pre-computation pipeline."""
import pytest
from src.analyst.schema import TradeSignal


class TestDecisionMapping:
    def test_valid_decisions(self):
        """Verify all valid decisions are accepted."""
        for d in ["hold", "buy", "sell"]:
            s = TradeSignal(reasoning="test", decision=d)
            assert s.decision == d

    def test_invalid_decision_rejected(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="test", decision="panic")
```

### Acceptance Criteria

- [ ] `precompute_signals()` supports resumption (re-running skips already-computed headlines)
- [ ] `precompute_signals()` produces `data/processed/precomputed_signals.json`
- [ ] Validation script confirms all cached signals produce valid TradeSignal objects
- [ ] All tests pass: `pytest tests/test_analyst_conversion.py -v`

### Files to Create/Modify

- `src/analyst/precompute.py` (create)
- `scripts/run_precompute.py` (create)
- `tests/test_analyst_conversion.py` (create)

### Dependencies

- P3-T1 must be complete (`schema.py`, `client.py` exist)
- `data/processed/headlines.json` must exist (from Phase 1)

### Human Checkpoint

- Run `scripts/run_precompute.py` and verify it completes. Check a few entries in `data/processed/precomputed_signals.json` — do the decisions match what you'd expect from the headlines? If most headlines produce "hold", the LLM may not be following the prompt properly.

---

## P3-T3: Analyst Validation and Consistency Testing

**Estimated time:** ~2 hours
**Dependencies:** P3-T1 and P3-T2 must be complete

### Context

Before integrating the Analyst into the C-Gate pipeline, we need to validate its behavior along three axes:

1. **Schema robustness**: Edge cases in schema validation and error handling.
2. **Directional accuracy**: Does the LLM produce sensible signals? We test this against FinancialPhraseBank (a labeled dataset of financial sentences with positive/negative/neutral sentiment).
3. **Preview of C-Gate dynamics**: By comparing the Analyst's decision against the Executor's policy distribution on overlapping dates, we can preview the Δ = 1 - π_RL(d_LLM) distribution and estimate how the C-Gate will behave.

**Already available:**
- `src/analyst/schema.py` — TradeSignal Pydantic model
- `src/analyst/client.py` — AnalystClient with OllamaBackend
- `data/processed/precomputed_signals.json` — pre-computed signals from P3-T2
- `data/raw/` — FinancialPhraseBank dataset (raw text files with sentiment labels)
- `experiments/executor/best_model/model.zip` — frozen PPO model from Phase 2
- `src/executor/policy.py` — `get_policy_distribution(model, obs) → np.array` from Phase 2
- `data/processed/aapl_features.parquet` — numeric features for the Executor

### Objective

Validate the Analyst agent's accuracy, consistency, and preview its interaction with the Executor via divergence analysis.

### Detailed Instructions

#### Step 1: Extend tests in `tests/test_analyst.py`

Create `tests/test_analyst.py` (a comprehensive integration test file):

```python
"""Comprehensive Analyst agent tests."""
import json
import pytest
from unittest.mock import MagicMock
from src.analyst.schema import TradeSignal
from src.analyst.client import AnalystClient, FALLBACK_SIGNAL, LLMBackend


class TestSchemaEdgeCases:
    def test_whitespace_reasoning(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="   \n\t  ", decision="hold")

    def test_long_reasoning(self):
        long_text = "x" * 10000
        s = TradeSignal(reasoning=long_text, decision="sell")
        assert len(s.reasoning) == 10000

    def test_json_roundtrip(self):
        original = TradeSignal(reasoning="test logic", decision="buy")
        serialized = original.model_dump_json()
        restored = TradeSignal.model_validate_json(serialized)
        assert restored.decision == original.decision
        assert restored.reasoning == original.reasoning


class MockBrokenJsonBackend(LLMBackend):
    """Returns invalid JSON."""
    def call(self, system_prompt, messages):
        return "This is not valid JSON at all!"


class TestClientErrorHandling:
    def test_invalid_json_falls_back(self):
        client = AnalystClient(
            backend=MockBrokenJsonBackend(),
            max_retries=2,
            include_few_shot=False
        )
        signal = client.analyze("test", "AAPL", "2024-01-01")
        assert signal.decision == FALLBACK_SIGNAL.decision
        assert signal.reasoning == FALLBACK_SIGNAL.reasoning
```

#### Step 2: Create `scripts/validate_analyst.py` for consistency and accuracy testing

```python
"""Analyst validation: accuracy vs FinancialPhraseBank, consistency, and Δ preview."""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Part 1: Accuracy vs FinancialPhraseBank
# ------------------------------------------------------------------
def load_phrasebank(path: str = "data/raw/", max_samples: int = 50) -> list[dict]:
    """
    Load FinancialPhraseBank sentences with labels.

    Expects a file with lines like:
        "sentence text"@label
    where label ∈ {positive, negative, neutral}.

    Adjust parsing logic to match your actual file format.
    """
    samples = []
    # Adjust this path and parsing to match the actual FinancialPhraseBank format
    fpb_path = Path(path)
    fpb_files = list(fpb_path.glob("*FinancialPhraseBank*")) + list(fpb_path.glob("*Sentences*"))

    if not fpb_files:
        logger.warning(f"No FinancialPhraseBank files found in {path}. Skipping accuracy test.")
        return []

    fpb_file = fpb_files[0]
    logger.info(f"Loading FinancialPhraseBank from {fpb_file}")

    with open(fpb_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                parts = line.rsplit("@", 1)
                if len(parts) == 2:
                    sentence, label = parts[0].strip(), parts[1].strip().lower()
                    if label in ("positive", "negative", "neutral"):
                        samples.append({"sentence": sentence, "label": label})

    # Sample evenly across labels
    from itertools import groupby
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    selected = []
    per_label = max_samples // 3
    for label, items in by_label.items():
        selected.extend(items[:per_label])

    logger.info(f"Selected {len(selected)} samples: {Counter(s['label'] for s in selected)}")
    return selected


def map_label_to_decision(label: str) -> str:
    """Map FinancialPhraseBank labels to our decision space."""
    return {"positive": "buy", "negative": "sell", "neutral": "hold"}[label]


def run_accuracy_test():
    """Test Analyst accuracy against FinancialPhraseBank ground truth."""
    from src.analyst.client import AnalystClient, OllamaBackend

    samples = load_phrasebank()
    if not samples:
        return

    backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)

    correct = 0
    results = []

    for sample in samples:
        signal = client.analyze(sample["sentence"], "TEST", "2024-01-01")
        expected = map_label_to_decision(sample["label"])
        is_correct = signal.decision == expected
        correct += int(is_correct)
        results.append({
            "sentence": sample["sentence"][:80],
            "label": sample["label"],
            "expected": expected,
            "predicted": signal.decision,
            "correct": is_correct,
        })

    accuracy = correct / len(results) if results else 0
    logger.info(f"Accuracy: {correct}/{len(results)} = {accuracy:.2%}")

    # Log to W&B if available
    try:
        import wandb
        wandb.init(project="robust-trinity", name="analyst-validation", reinit=True)
        wandb.log({"analyst/accuracy": accuracy, "analyst/n_samples": len(results)})
        wandb.log({"analyst/results_table": wandb.Table(
            columns=list(results[0].keys()),
            data=[list(r.values()) for r in results]
        )})
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
        return

    backend = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    client = AnalystClient(backend=backend, max_retries=3, include_few_shot=True)

    # Collect decisions across runs
    all_decisions = {i: [] for i in range(n_runs)}
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

    mean_kappa = np.mean(kappas)
    logger.info(f"Mean Cohen's kappa: {mean_kappa:.3f}")

    try:
        import wandb
        wandb.log({"analyst/mean_kappa": mean_kappa})
    except Exception:
        pass


# ------------------------------------------------------------------
# Part 3: Δ Preview — Analyst vs Executor Divergence
# ------------------------------------------------------------------
# Action mapping: must match Executor's action space
DECISION_TO_INDEX = {"hold": 0, "buy": 1, "sell": 2}


def run_delta_preview():
    """
    Compare Analyst decisions against Executor policy distributions
    on overlapping dates. Computes Δ = 1 - π_RL(d_LLM | s_t).
    This previews C-Gate behavior.
    """
    from src.analyst.schema import TradeSignal

    # Load precomputed signals
    signals_path = "data/processed/precomputed_signals.json"
    if not Path(signals_path).exists():
        logger.error(f"Precomputed signals not found at {signals_path}. Run P3-T2 first.")
        return

    with open(signals_path, "r") as f:
        signals_cache = json.load(f)

    # Load executor model and features
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from src.executor.policy import get_policy_distribution

        model = PPO.load("experiments/executor/best_model/model.zip")
        features = pd.read_parquet("data/processed/aapl_features.parquet")
    except Exception as e:
        logger.error(f"Could not load Executor model/features: {e}")
        return

    # Match by date
    deltas = []
    dates = []

    for key, entry in signals_cache.items():
        date = entry["date"]
        signal = TradeSignal(
            reasoning=entry["reasoning"],
            decision=entry["decision"],
        )
        action_index = DECISION_TO_INDEX[signal.decision]

        # Find matching observation in features
        if date in features.index or date in features.get("date", pd.Series()).values:
            # Get the observation row — adapt indexing to your features format
            try:
                if "date" in features.columns:
                    obs_row = features[features["date"] == date].iloc[0]
                else:
                    obs_row = features.loc[date]

                obs = obs_row.values.astype(np.float32)
                pi_rl = get_policy_distribution(model, obs)

                # Δ = 1 - π_RL(d_LLM | s_t)
                delta = 1.0 - pi_rl[action_index]
                deltas.append(delta)
                dates.append(date)
            except Exception as e:
                logger.debug(f"Skipping date {date}: {e}")
                continue

    if not deltas:
        logger.warning("No overlapping dates found between signals and features.")
        return

    deltas = np.array(deltas)
    logger.info(f"Δ statistics over {len(deltas)} dates:")
    logger.info(f"  Mean:   {deltas.mean():.4f}")
    logger.info(f"  Median: {np.median(deltas):.4f}")
    logger.info(f"  Std:    {deltas.std():.4f}")
    logger.info(f"  Min:    {deltas.min():.4f}")
    logger.info(f"  Max:    {deltas.max():.4f}")

    # Regime counts with initial thresholds
    tau_low, tau_high = 0.1, 0.4
    agreement = np.sum(deltas <= tau_low) / len(deltas)
    ambiguity = np.sum((deltas > tau_low) & (deltas <= tau_high)) / len(deltas)
    conflict = np.sum(deltas > tau_high) / len(deltas)

    logger.info(f"  Agreement (Δ ≤ {tau_low}): {agreement:.1%}")
    logger.info(f"  Ambiguity ({tau_low} < Δ ≤ {tau_high}): {ambiguity:.1%}")
    logger.info(f"  Conflict  (Δ > {tau_high}): {conflict:.1%}")

    # Log to W&B
    try:
        import wandb
        wandb.init(project="robust-trinity", name="delta-preview", reinit=True)
        wandb.log({
            "delta/mean": deltas.mean(),
            "delta/median": np.median(deltas),
            "delta/std": deltas.std(),
            "delta/pct_agreement": agreement,
            "delta/pct_ambiguity": ambiguity,
            "delta/pct_conflict": conflict,
        })
        # Histogram
        wandb.log({"delta/histogram": wandb.Histogram(deltas, num_bins=50)})
    except Exception as e:
        logger.warning(f"W&B logging failed: {e}")

    return deltas


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
```

### Acceptance Criteria

- [ ] All unit tests pass: `pytest tests/test_analyst.py -v`
- [ ] Accuracy test runs against FinancialPhraseBank — accuracy should be ≥ 50% (random would be ~33%)
- [ ] Consistency test produces Cohen's kappa ≥ 0.7 (substantial agreement) at temperature=0
- [ ] Δ preview produces a histogram showing all three regimes
- [ ] Results are logged to W&B under project "robust-trinity"

### Files to Create/Modify

- `tests/test_analyst.py` (create)
- `scripts/validate_analyst.py` (create)

### Dependencies

- P3-T1 and P3-T2 must be complete
- `data/processed/precomputed_signals.json` must exist (from P3-T2)
- `experiments/executor/best_model/model.zip` must exist (from Phase 2)
- `data/raw/` must contain FinancialPhraseBank data (from Phase 1)

### Human Checkpoint

- Review the Δ distribution histogram. If >50% of timesteps are in the "conflict" regime under benign conditions, the agents are too misaligned — investigate whether the Analyst is producing sensible signals. The expected distribution under benign conditions is: majority agreement/ambiguity, minority conflict.
- Spot-check 5–10 Analyst signals against their headlines: does "Apple beats earnings" → buy? Does "FDA rejects drug" → sell? If the Analyst is directionally wrong on obvious cases, the prompt or LLM model needs tuning.
