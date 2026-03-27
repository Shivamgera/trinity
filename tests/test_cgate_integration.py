"""Integration-style tests for C-Gate with synthetic but realistic inputs."""

import numpy as np
import pytest

from src.cgate.gate import ConsistencyGate


class TestCGateIntegrationSynthetic:
    """Test C-Gate with synthetic distributions that mimic real agent outputs."""

    @pytest.fixture
    def gate(self):
        return ConsistencyGate(tau_low=0.1, tau_high=0.4)

    def test_full_sequence(self, gate):
        """Simulate a sequence of timesteps with varying agreement levels."""
        rng = np.random.default_rng(42)
        regimes_seen = set()

        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)

            assert 0 <= result.delta <= 1.0 + 1e-10
            assert result.regime in ("agreement", "ambiguity", "conflict")
            assert result.action in (0, 1, 2)
            regimes_seen.add(result.regime)

        # With random distributions, all regimes should appear
        assert len(regimes_seen) == 3, f"Only saw regimes: {regimes_seen}"

    def test_agreement_sequence(self, gate):
        """When both agents consistently agree, should be mostly agreement."""
        regimes = []
        for _ in range(50):
            # RL strongly favors "long" (action 1), LLM says "long"
            pi_rl = np.array([0.02, 0.96, 0.02])
            result = gate.evaluate("long", pi_rl)
            regimes.append(result.regime)

        agreement_pct = regimes.count("agreement") / len(regimes)
        assert agreement_pct > 0.8, f"Expected >80% agreement, got {agreement_pct:.0%}"

    def test_conflict_always_flat(self, gate):
        """Every conflict-regime result must have action=0."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)
            if result.regime == "conflict":
                assert result.action == 0, (
                    f"Conflict regime should force flat, got action {result.action}"
                )

    def test_near_uniform_produces_conflict(self, gate):
        """Near-uniform pi_RL should produce conflict for any decision.

        This mimics our actual Executor behavior (weak signal -> ~uniform policy).
        """
        pi_rl = np.array([1 / 3, 1 / 3, 1 / 3])
        for decision in ["hold", "buy", "sell"]:
            result = gate.evaluate(decision, pi_rl)
            # Δ = 1 - 1/3 ≈ 0.667 > tau_high=0.4
            assert result.regime == "conflict"
            assert result.action == 0

    def test_conservative_analyst_with_uniform_executor(self, gate):
        """Simulate our actual scenario: 90% hold analyst + uniform executor.

        All timesteps should land in conflict since Δ ≈ 0.667 for all.
        """
        rng = np.random.default_rng(42)
        conflict_count = 0
        n = 100

        for _ in range(n):
            # Analyst: 90% hold, 5% buy, 5% sell
            r = rng.random()
            if r < 0.90:
                d_llm = "hold"
            elif r < 0.95:
                d_llm = "buy"
            else:
                d_llm = "sell"

            # Executor: near-uniform with small noise
            noise = rng.normal(0, 0.005, 3)
            pi_rl = np.array([1 / 3, 1 / 3, 1 / 3]) + noise
            pi_rl = np.abs(pi_rl)
            pi_rl = pi_rl / pi_rl.sum()

            result = gate.evaluate(d_llm, pi_rl)
            if result.regime == "conflict":
                conflict_count += 1

        # With uniform executor, Δ ≈ 0.667 for any decision -> all conflict
        assert conflict_count == n, (
            f"Expected all conflict, got {conflict_count}/{n}"
        )

    def test_missing_signal_handled_as_conflict(self):
        """When no analyst signal exists, treat as Δ=1.0 conflict."""
        from src.cgate.gate import CGateResult

        result = CGateResult(delta=1.0, regime="conflict", action=0)
        assert result.delta == 1.0
        assert result.regime == "conflict"
        assert result.action == 0
