"""Tests for TradeSignal schema and AnalystClient."""

import json

import pytest

from src.analyst.client import (
    FALLBACK_SIGNAL,
    AnalystClient,
    LLMBackend,
)
from src.analyst.schema import TradeSignal


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestTradeSignal:
    def test_valid_signal(self):
        s = TradeSignal(reasoning="test reasoning", decision="buy")
        assert s.decision == "buy"

    def test_all_valid_decisions(self):
        for d in ["hold", "buy", "sell"]:
            s = TradeSignal(reasoning="test", decision=d)
            assert s.decision == d

    def test_invalid_decision(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="test", decision="panic")

    def test_empty_reasoning_rejected(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="", decision="hold")

    def test_whitespace_reasoning_rejected(self):
        with pytest.raises(Exception):
            TradeSignal(reasoning="   \n\t  ", decision="hold")

    def test_missing_fields(self):
        with pytest.raises(Exception):
            TradeSignal(decision="buy")  # type: ignore[call-arg]

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


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


class MockValidBackend(LLMBackend):
    def call(self, system_prompt: str, messages: list[dict]) -> str:
        return json.dumps({"reasoning": "Looks bullish", "decision": "buy"})


class MockFailingBackend(LLMBackend):
    def call(self, system_prompt: str, messages: list[dict]) -> str:
        raise ConnectionError("API down")


class MockBrokenJsonBackend(LLMBackend):
    """Returns invalid JSON."""

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        return "This is not valid JSON at all!"


class MockInvalidSchemaBackend(LLMBackend):
    """Returns valid JSON but invalid schema."""

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        return json.dumps({"reasoning": "good", "decision": "panic"})


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


class TestAnalystClient:
    def test_successful_analysis(self):
        client = AnalystClient(backend=MockValidBackend(), include_few_shot=False)
        signal = client.analyze("AAPL beats earnings", "AAPL", "2024-01-15")
        assert signal.decision == "buy"
        assert signal.reasoning == "Looks bullish"

    def test_fallback_on_connection_error(self):
        client = AnalystClient(
            backend=MockFailingBackend(), max_retries=2, include_few_shot=False
        )
        signal = client.analyze("test headline", "AAPL", "2024-01-15")
        assert signal.decision == FALLBACK_SIGNAL.decision
        assert signal.reasoning == FALLBACK_SIGNAL.reasoning

    def test_fallback_on_invalid_json(self):
        client = AnalystClient(
            backend=MockBrokenJsonBackend(), max_retries=2, include_few_shot=False
        )
        signal = client.analyze("test", "AAPL", "2024-01-01")
        assert signal.decision == FALLBACK_SIGNAL.decision

    def test_fallback_on_invalid_schema(self):
        client = AnalystClient(
            backend=MockInvalidSchemaBackend(), max_retries=2, include_few_shot=False
        )
        signal = client.analyze("test", "AAPL", "2024-01-01")
        assert signal.decision == FALLBACK_SIGNAL.decision


# ---------------------------------------------------------------------------
# Precompute tests
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_headline_hash_deterministic(self):
        from src.analyst.precompute import headline_hash

        h1 = headline_hash("Apple beats earnings", "AAPL", "2024-01-15")
        h2 = headline_hash("Apple beats earnings", "AAPL", "2024-01-15")
        assert h1 == h2

    def test_headline_hash_different_for_different_inputs(self):
        from src.analyst.precompute import headline_hash

        h1 = headline_hash("Apple beats", "AAPL", "2024-01-15")
        h2 = headline_hash("Apple misses", "AAPL", "2024-01-15")
        assert h1 != h2

    def test_precompute_signals(self, tmp_path):
        from src.analyst.precompute import precompute_signals

        headlines = [
            {
                "headline": "AAPL beats earnings",
                "ticker": "AAPL",
                "date": "2024-01-15",
            },
            {
                "headline": "AAPL misses revenue",
                "ticker": "AAPL",
                "date": "2024-01-16",
            },
        ]

        client = AnalystClient(
            backend=MockValidBackend(), include_few_shot=False, max_retries=1
        )
        output_path = str(tmp_path / "cache.json")

        cache = precompute_signals(
            headlines=headlines,
            client=client,
            output_path=output_path,
            delay=0.0,
            save_every=1,
        )

        assert len(cache) == 2
        for entry in cache.values():
            assert entry["decision"] == "buy"

    def test_precompute_resumption(self, tmp_path):
        """Running precompute twice should not re-process existing entries."""
        from src.analyst.precompute import precompute_signals

        headlines = [
            {
                "headline": "test headline",
                "ticker": "AAPL",
                "date": "2024-01-15",
            },
        ]

        call_count = 0

        class CountingBackend(LLMBackend):
            def call(self, system_prompt: str, messages: list[dict]) -> str:
                nonlocal call_count
                call_count += 1
                return json.dumps(
                    {"reasoning": "test", "decision": "hold"}
                )

        client = AnalystClient(
            backend=CountingBackend(), include_few_shot=False, max_retries=1
        )
        output_path = str(tmp_path / "cache.json")

        # First run
        precompute_signals(
            headlines=headlines,
            client=client,
            output_path=output_path,
            delay=0.0,
        )
        assert call_count == 1

        # Second run: should skip
        precompute_signals(
            headlines=headlines,
            client=client,
            output_path=output_path,
            delay=0.0,
        )
        assert call_count == 1  # not incremented
