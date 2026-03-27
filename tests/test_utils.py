"""Tests for utility modules — seed reproducibility and data stubs."""

import random

import numpy as np
import torch

from src.utils.seed import set_global_seed


class TestSeedReproducibility:
    """Verify that set_global_seed produces identical sequences."""

    def test_numpy_reproducibility(self):
        set_global_seed(42)
        a1 = np.random.rand(10)
        set_global_seed(42)
        a2 = np.random.rand(10)
        np.testing.assert_array_equal(a1, a2)

    def test_torch_reproducibility(self):
        set_global_seed(42)
        t1 = torch.randn(10)
        set_global_seed(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_random_reproducibility(self):
        set_global_seed(42)
        r1 = [random.random() for _ in range(10)]
        set_global_seed(42)
        r2 = [random.random() for _ in range(10)]
        assert r1 == r2

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a1 = np.random.rand(10)
        set_global_seed(99)
        a2 = np.random.rand(10)
        assert not np.array_equal(a1, a2)


class TestDataLoading:
    """Verify data loading functions work with real data."""

    def test_load_numeric_features(self):
        from src.utils.data import load_numeric_features

        df = load_numeric_features()
        assert len(df) > 200

    def test_load_headlines(self):
        from src.utils.data import load_headlines

        headlines = load_headlines()
        assert len(headlines) > 400
