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
