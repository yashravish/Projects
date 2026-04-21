"""Unit tests for the constant-product AMM."""

import numpy as np
import pytest

from execsim.simulator.amm import AMMPool, create_pool, rebalance_pool


class TestAMMPool:
    """Tests for AMMPool pricing and swap math."""

    @pytest.fixture()
    def pool(self):
        return AMMPool(reserve_x=10000.0, reserve_y=1000000.0, fee_bps=30.0)

    def test_spot_price(self, pool):
        """Spot price = reserve_y / reserve_x."""
        assert pool.spot_price == 100.0

    def test_k_invariant(self, pool):
        assert pool.k == 10000.0 * 1000000.0

    def test_quote_to_buy_positive(self, pool):
        dy = pool.quote_to_buy(10.0)
        assert dy > 0

    def test_buy_price_exceeds_spot(self, pool):
        """Buying pushes price up; average price > spot for finite qty."""
        exec_p = pool.exec_price_buy(10.0)
        assert exec_p > pool.spot_price

    def test_sell_price_below_spot(self, pool):
        """Selling pushes price down; average price < spot for finite qty."""
        exec_p = pool.exec_price_sell(10.0)
        assert exec_p < pool.spot_price

    def test_small_trade_near_spot(self, pool):
        """For very small trades, exec price should be close to spot."""
        exec_p = pool.exec_price_buy(0.001)
        assert abs(exec_p - pool.spot_price) / pool.spot_price < 0.01

    def test_cannot_buy_entire_reserve(self, pool):
        with pytest.raises(ValueError, match="must be < reserve_x"):
            pool.quote_to_buy(10000.0)

    def test_sell_received_positive(self, pool):
        dy = pool.quote_received_sell(10.0)
        assert dy > 0

    def test_invalid_reserve(self):
        with pytest.raises(ValueError, match="reserve_x must be > 0"):
            AMMPool(reserve_x=0, reserve_y=100, fee_bps=0)

    def test_fee_effect(self):
        """Higher fee means worse execution for the trader."""
        pool_low = AMMPool(10000.0, 1000000.0, fee_bps=0.0)
        pool_high = AMMPool(10000.0, 1000000.0, fee_bps=100.0)
        # Buying: higher fee means higher cost
        assert pool_high.exec_price_buy(10.0) > pool_low.exec_price_buy(10.0)
        # Selling: higher fee means lower received
        assert pool_high.exec_price_sell(10.0) < pool_low.exec_price_sell(10.0)


class TestCreatePool:
    def test_creates_correct_spot(self):
        pool = create_pool(initial_price=100.0, reserve_x=10000.0, fee_bps=30.0)
        assert abs(pool.spot_price - 100.0) < 1e-10

    def test_invalid_price(self):
        with pytest.raises(ValueError, match="initial_price must be > 0"):
            create_pool(initial_price=0, reserve_x=10000.0, fee_bps=30.0)


class TestRebalancePool:
    def test_preserves_k(self):
        pool = create_pool(100.0, 10000.0, 30.0)
        k_before = pool.k
        rng = np.random.default_rng(42)
        new_pool = rebalance_pool(pool, 110.0, 0.0, rng)
        assert abs(new_pool.k - k_before) / k_before < 1e-12

    def test_tracks_target_price(self):
        pool = create_pool(100.0, 10000.0, 30.0)
        rng = np.random.default_rng(42)
        new_pool = rebalance_pool(pool, 150.0, 0.0, rng)
        assert abs(new_pool.spot_price - 150.0) < 1e-6

    def test_deterministic(self):
        pool = create_pool(100.0, 10000.0, 30.0)
        p1 = rebalance_pool(pool, 110.0, 0.001, np.random.default_rng(42))
        p2 = rebalance_pool(pool, 110.0, 0.001, np.random.default_rng(42))
        assert p1.spot_price == p2.spot_price
