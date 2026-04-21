"""Unit tests for the order book model."""

import numpy as np
import pytest

from execsim.simulator.orderbook import BookLevel, OrderBook, build_order_book


class TestOrderBook:
    """Tests for OrderBook properties and fill simulation."""

    @pytest.fixture()
    def simple_book(self):
        return OrderBook(
            bids=[BookLevel(100.0, 5.0), BookLevel(99.0, 10.0)],
            asks=[BookLevel(101.0, 5.0), BookLevel(102.0, 10.0)],
        )

    def test_best_bid(self, simple_book):
        assert simple_book.best_bid == 100.0

    def test_best_ask(self, simple_book):
        assert simple_book.best_ask == 101.0

    def test_mid(self, simple_book):
        assert simple_book.mid == 100.5

    def test_spread(self, simple_book):
        assert simple_book.spread == 1.0

    def test_spread_nonnegative(self, simple_book):
        assert simple_book.spread >= 0

    def test_vwap_buy_single_level(self, simple_book):
        vwap, filled = simple_book.vwap_buy(3.0)
        assert filled == 3.0
        assert vwap == 101.0  # all from first level

    def test_vwap_buy_across_levels(self, simple_book):
        vwap, filled = simple_book.vwap_buy(10.0)
        assert filled == 10.0
        # 5 at 101 + 5 at 102 = 505 + 510 = 1015, vwap = 101.5
        assert abs(vwap - 101.5) < 1e-10

    def test_vwap_buy_exceeds_depth(self, simple_book):
        vwap, filled = simple_book.vwap_buy(20.0)
        assert filled == 15.0  # only 15 available
        assert vwap > 0

    def test_vwap_sell_single_level(self, simple_book):
        vwap, filled = simple_book.vwap_sell(3.0)
        assert filled == 3.0
        assert vwap == 100.0

    def test_vwap_buy_invalid_qty(self):
        book = OrderBook(
            bids=[BookLevel(100.0, 5.0)],
            asks=[BookLevel(101.0, 5.0)],
        )
        with pytest.raises(ValueError, match="qty must be > 0"):
            book.vwap_buy(-1.0)


class TestBuildOrderBook:
    """Tests for build_order_book."""

    def test_returns_correct_levels(self):
        rng = np.random.default_rng(42)
        book = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, rng)
        assert len(book.bids) == 5
        assert len(book.asks) == 5

    def test_bids_descending(self):
        rng = np.random.default_rng(42)
        book = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, rng)
        for i in range(len(book.bids) - 1):
            assert book.bids[i].price >= book.bids[i + 1].price

    def test_asks_ascending(self):
        rng = np.random.default_rng(42)
        book = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, rng)
        for i in range(len(book.asks) - 1):
            assert book.asks[i].price <= book.asks[i + 1].price

    def test_spread_positive(self):
        rng = np.random.default_rng(42)
        book = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, rng)
        assert book.spread > 0

    def test_deterministic(self):
        b1 = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, np.random.default_rng(42))
        b2 = build_order_book(100.0, 5.0, 5, 0.01, 1.0, 10.0, np.random.default_rng(42))
        assert b1.best_bid == b2.best_bid
        assert b1.best_ask == b2.best_ask

    def test_invalid_mid_price(self):
        with pytest.raises(ValueError, match="mid_price must be > 0"):
            build_order_book(0, 5.0, 5, 0.01, 1.0, 10.0, np.random.default_rng(0))
