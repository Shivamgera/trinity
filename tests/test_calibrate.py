"""Tests for the C-Gate threshold calibration module."""

import numpy as np
import pytest

from src.cgate.calibrate import calibrate_thresholds, collect_val_deltas


class TestCalibrateThresholds:
    """Unit tests for calibrate_thresholds (using live frozen models + signals)."""

    # Use only validated seeds from selection.json (1024/4096 have different obs dims)
    VALID_SEEDS = [123, 7777, 2048, 789]

    @pytest.fixture(scope="class")
    def cal_result(self):
        """Run calibration once and cache for the class."""
        return calibrate_thresholds(
            temperature=0.01,
            low_percentile=20.0,
            high_percentile=80.0,
            seeds=TestCalibrateThresholds.VALID_SEEDS,
        )

    def test_returns_dict_with_required_keys(self, cal_result):
        required = {
            "temperature",
            "low_percentile",
            "high_percentile",
            "tau_low",
            "tau_high",
            "n_samples",
            "pct_agreement",
            "pct_ambiguity",
            "pct_conflict",
        }
        assert required.issubset(cal_result.keys())

    def test_tau_ordering(self, cal_result):
        assert 0 < cal_result["tau_low"] < cal_result["tau_high"] < 1

    def test_regimes_sum_to_one(self, cal_result):
        total = (
            cal_result["pct_agreement"]
            + cal_result["pct_ambiguity"]
            + cal_result["pct_conflict"]
        )
        assert abs(total - 1.0) < 0.01  # rounding tolerance

    def test_no_degenerate_regime(self, cal_result):
        """Each regime should have at least some representation."""
        assert cal_result["pct_agreement"] > 0.05, "Agreement regime too small"
        assert cal_result["pct_ambiguity"] > 0.20, "Ambiguity regime too small"
        assert cal_result["pct_conflict"] > 0.05, "Conflict regime too small"

    def test_sufficient_samples(self, cal_result):
        """Should have pooled data from multiple seeds."""
        # 4 seeds × ~113 val steps = ~452
        assert cal_result["n_samples"] >= 400

    def test_temperature_recorded(self, cal_result):
        assert cal_result["temperature"] == 0.01


class TestCollectValDeltas:
    """Tests for collect_val_deltas."""

    def test_returns_numpy_array(self):
        deltas = collect_val_deltas(temperature=0.01, seeds=[123])
        assert isinstance(deltas, np.ndarray)

    def test_deltas_bounded(self):
        deltas = collect_val_deltas(temperature=0.01, seeds=[123])
        assert np.all(deltas >= 0.0)
        assert np.all(deltas <= 1.0)

    def test_has_spread_with_temperature(self):
        """With T=0.01, deltas should have meaningful variance."""
        deltas = collect_val_deltas(temperature=0.01, seeds=[123])
        assert deltas.std() > 0.05, f"Delta std too low: {deltas.std():.4f}"

    def test_no_spread_without_temperature(self):
        """With T=1.0, deltas should be tightly clustered around 0.667."""
        deltas = collect_val_deltas(temperature=1.0, seeds=[123])
        assert deltas.std() < 0.02, f"Delta std unexpectedly high: {deltas.std():.4f}"

    def test_missing_seeds_raises(self):
        with pytest.raises(FileNotFoundError):
            collect_val_deltas(frozen_dir="nonexistent_dir", seeds=[999])
