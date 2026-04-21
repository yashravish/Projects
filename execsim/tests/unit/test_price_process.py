"""Unit tests for the GBM price process."""

import numpy as np
import pytest

from execsim.simulator.price_process import generate_mid_prices


class TestGenerateMidPrices:
    """Tests for generate_mid_prices."""

    def test_length(self):
        rng = np.random.default_rng(42)
        prices = generate_mid_prices(100, 100.0, 0.0, 0.02, 0.1, rng)
        assert len(prices) == 100

    def test_initial_price(self):
        rng = np.random.default_rng(42)
        prices = generate_mid_prices(100, 50.0, 0.0, 0.02, 0.1, rng)
        assert prices[0] == 50.0

    def test_all_positive(self):
        """GBM guarantees strictly positive prices."""
        rng = np.random.default_rng(99)
        prices = generate_mid_prices(5000, 100.0, 0.0, 0.05, 0.1, rng)
        assert np.all(prices > 0)

    def test_deterministic(self):
        """Same seed produces identical prices."""
        p1 = generate_mid_prices(100, 100.0, 0.0, 0.02, 0.1, np.random.default_rng(7))
        p2 = generate_mid_prices(100, 100.0, 0.0, 0.02, 0.1, np.random.default_rng(7))
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = generate_mid_prices(100, 100.0, 0.0, 0.02, 0.1, np.random.default_rng(1))
        p2 = generate_mid_prices(100, 100.0, 0.0, 0.02, 0.1, np.random.default_rng(2))
        assert not np.array_equal(p1, p2)

    def test_single_step(self):
        rng = np.random.default_rng(42)
        prices = generate_mid_prices(1, 100.0, 0.0, 0.02, 0.1, rng)
        assert len(prices) == 1
        assert prices[0] == 100.0

    def test_zero_vol_constant_price(self):
        """With zero drift and zero vol, price stays constant."""
        rng = np.random.default_rng(42)
        prices = generate_mid_prices(100, 100.0, 0.0, 0.0, 0.1, rng)
        np.testing.assert_array_almost_equal(prices, 100.0)

    def test_invalid_initial_price(self):
        with pytest.raises(ValueError, match="initial_price must be > 0"):
            generate_mid_prices(10, -1.0, 0.0, 0.02, 0.1, np.random.default_rng(0))

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be >= 1"):
            generate_mid_prices(0, 100.0, 0.0, 0.02, 0.1, np.random.default_rng(0))
