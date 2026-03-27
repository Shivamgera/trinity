"""Comprehensive tests for the Consistency Gate."""

import numpy as np
import pytest

from src.cgate.divergence import compute_delta
from src.cgate.gate import CGateResult, ConsistencyGate


class TestComputeDelta:
    """Tests for the compute_delta function."""

    def test_perfect_agreement(self):
        """When RL assigns probability 1.0 to LLM's action, Δ = 0."""
        delta = compute_delta("buy", np.array([0.0, 1.0, 0.0]))
        assert delta == 0.0

    def test_perfect_disagreement(self):
        """When RL assigns probability 0.0 to LLM's action, Δ = 1."""
        delta = compute_delta("buy", np.array([0.5, 0.0, 0.5]))
        assert delta == 1.0

    def test_moderate_agreement(self):
        """When RL assigns moderate probability, Δ is moderate."""
        delta = compute_delta("buy", np.array([0.2, 0.6, 0.2]))
        assert abs(delta - 0.4) < 1e-10

    def test_uniform_distribution(self):
        """Uniform π_RL → Δ = 1 - 1/3 ≈ 0.667."""
        delta = compute_delta("hold", np.array([1 / 3, 1 / 3, 1 / 3]))
        assert abs(delta - 2 / 3) < 1e-10

    def test_bounded_zero_one(self):
        """Δ should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            delta = compute_delta(d_llm, pi_rl)
            assert 0.0 <= delta <= 1.0 + 1e-10, f"Δ={delta} out of bounds"

    def test_string_and_int_equivalent(self):
        """String 'buy' and int 1 should give the same result."""
        pi_rl = np.array([0.2, 0.5, 0.3])
        assert compute_delta("buy", pi_rl) == compute_delta(1, pi_rl)
        assert compute_delta("hold", pi_rl) == compute_delta(0, pi_rl)
        assert compute_delta("sell", pi_rl) == compute_delta(2, pi_rl)

    def test_alternative_names(self):
        """'flat'/'long'/'short' should map to 0/1/2."""
        pi_rl = np.array([0.3, 0.5, 0.2])
        assert compute_delta("flat", pi_rl) == compute_delta("hold", pi_rl)
        assert compute_delta("long", pi_rl) == compute_delta("buy", pi_rl)
        assert compute_delta("short", pi_rl) == compute_delta("sell", pi_rl)

    def test_invalid_decision_raises(self):
        """Unknown decision string should raise ValueError."""
        with pytest.raises(ValueError):
            compute_delta("unknown", np.array([0.3, 0.4, 0.3]))

    def test_invalid_pi_rl_shape(self):
        """Wrong shape pi_rl should raise ValueError."""
        with pytest.raises(ValueError):
            compute_delta("buy", np.array([0.5, 0.5]))

    def test_invalid_pi_rl_sum(self):
        """pi_rl not summing to 1 should raise ValueError."""
        with pytest.raises(ValueError):
            compute_delta("buy", np.array([0.5, 0.5, 0.5]))

    def test_negative_pi_rl_raises(self):
        """Negative probabilities should raise ValueError."""
        with pytest.raises(ValueError):
            compute_delta("buy", np.array([-0.1, 0.6, 0.5]))

    def test_invalid_action_index_raises(self):
        """Action index out of range should raise ValueError."""
        with pytest.raises(ValueError):
            compute_delta(5, np.array([0.3, 0.4, 0.3]))

    def test_invalid_type_raises(self):
        """Non-str/int type should raise TypeError."""
        with pytest.raises(TypeError):
            compute_delta([1], np.array([0.3, 0.4, 0.3]))  # type: ignore[arg-type]


class TestConsistencyGate:
    """Tests for the C-Gate regime classification and action selection."""

    @pytest.fixture
    def gate(self):
        return ConsistencyGate(tau_low=0.1, tau_high=0.4)

    def test_agreement_regime(self, gate):
        """When RL strongly agrees with LLM → agreement, execute argmax(π_RL)."""
        # RL assigns 0.95 to "long" (action 1), LLM says "long"
        pi_rl = np.array([0.025, 0.95, 0.025])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "agreement"
        assert result.delta == pytest.approx(0.05, abs=1e-10)
        assert result.action == 1  # argmax of pi_rl

    def test_ambiguity_regime(self, gate):
        """When RL moderately disagrees with LLM → ambiguity."""
        # Δ = 1 - 0.75 = 0.25, which is in (0.1, 0.4]
        pi_rl = np.array([0.1, 0.75, 0.15])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "ambiguity"
        assert result.delta == pytest.approx(0.25, abs=1e-10)
        assert result.action == 1  # argmax of pi_rl (conservative mode applied by Guardian)

    def test_conflict_regime(self, gate):
        """When RL strongly disagrees with LLM → conflict, fail-safe to flat."""
        # RL assigns 0.05 to "long" (action 1), LLM says "long"
        pi_rl = np.array([0.6, 0.05, 0.35])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "conflict"
        assert result.delta == pytest.approx(0.95, abs=1e-10)
        assert result.action == 0  # flat (safe default)

    def test_conflict_forces_flat(self, gate):
        """In conflict regime, action must always be 0 (flat)."""
        # RL assigns 0.0 to "sell" (action 2)
        pi_rl = np.array([0.5, 0.5, 0.0])
        result = gate.evaluate("sell", pi_rl)
        assert result.regime == "conflict"
        assert result.action == 0

    def test_boundary_tau_low(self, gate):
        """Δ exactly at tau_low should be agreement (<=)."""
        # Δ = 1 - 0.9 = 0.1 = tau_low
        pi_rl = np.array([0.05, 0.9, 0.05])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "agreement"

    def test_boundary_tau_high(self, gate):
        """Δ exactly at tau_high should be ambiguity (<=)."""
        # Δ = 1 - 0.6 = 0.4 = tau_high
        pi_rl = np.array([0.2, 0.6, 0.2])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "ambiguity"

    def test_just_above_tau_high(self, gate):
        """Δ just above tau_high should be conflict."""
        # Δ = 1 - 0.59 = 0.41 > tau_high=0.4
        pi_rl = np.array([0.205, 0.59, 0.205])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "conflict"

    def test_delta_always_bounded(self, gate):
        """Δ should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            raw = rng.random(3)
            pi_rl = raw / raw.sum()
            d_llm = rng.choice(["hold", "buy", "sell"])
            result = gate.evaluate(d_llm, pi_rl)
            assert 0.0 <= result.delta <= 1.0 + 1e-10

    def test_custom_thresholds(self):
        """Gate respects custom threshold values."""
        gate = ConsistencyGate(tau_low=0.01, tau_high=0.05)
        # Δ = 1 - 0.5 = 0.5, which > 0.05 → conflict
        pi_rl = np.array([0.3, 0.5, 0.2])
        result = gate.evaluate("buy", pi_rl)
        assert result.regime == "conflict"

    def test_invalid_thresholds(self):
        """Invalid thresholds should raise ValueError."""
        with pytest.raises(ValueError):
            ConsistencyGate(tau_low=0.5, tau_high=0.3)  # low > high
        with pytest.raises(ValueError):
            ConsistencyGate(tau_low=-0.1, tau_high=0.5)  # negative

    def test_equal_thresholds(self):
        """tau_low == tau_high is valid — no ambiguity zone."""
        gate = ConsistencyGate(tau_low=0.3, tau_high=0.3)
        # Δ = 1 - 0.75 = 0.25 → agreement (<=0.3)
        pi_rl = np.array([0.1, 0.75, 0.15])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "agreement"
        # Δ = 1 - 0.5 = 0.5 → conflict (>0.3)
        pi_rl = np.array([0.25, 0.5, 0.25])
        result = gate.evaluate("long", pi_rl)
        assert result.regime == "conflict"

    def test_cgate_result_dataclass(self):
        """CGateResult is a proper dataclass with expected fields."""
        r = CGateResult(delta=0.5, regime="conflict", action=0)
        assert r.delta == 0.5
        assert r.regime == "conflict"
        assert r.action == 0
