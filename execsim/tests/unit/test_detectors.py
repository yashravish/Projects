"""Unit tests for detectors."""

import numpy as np
import pytest

from execsim.detectors.arbitrage import ArbitrageDetector
from execsim.detectors.base import MarketState, OpportunityKind, SideKind
from execsim.detectors.liquidation import LiquidationDetector
from execsim.detectors.stale_quote import StaleQuoteDetector
from execsim.simulator.amm import AMMPool
from execsim.simulator.orderbook import BookLevel, OrderBook


def _make_book(bid: float, ask: float, qty: float = 10.0) -> OrderBook:
    return OrderBook(
        bids=[BookLevel(bid, qty)],
        asks=[BookLevel(ask, qty)],
    )


def _make_state(
    true_mid: float,
    va_bid: float,
    va_ask: float,
    vb_bid: float,
    vb_ask: float,
    amm_price: float = 100.0,
    has_liq: bool = False,
) -> MarketState:
    amm_x = 10000.0
    amm_y = amm_price * amm_x
    return MarketState(
        step=0,
        true_mid=true_mid,
        venue_a_book=_make_book(va_bid, va_ask),
        venue_b_book=_make_book(vb_bid, vb_ask),
        amm_pool=AMMPool(amm_x, amm_y, fee_bps=30.0),
        has_liquidation=has_liq,
    )


class TestArbitrageDetector:
    def test_no_arb_normal_spread(self):
        det = ArbitrageDetector(threshold_bps=2.0, estimated_cost_bps=1.0)
        state = _make_state(100.0, 99.95, 100.05, 99.92, 100.08)
        opps = det.detect(state)
        assert len(opps) == 0

    def test_detects_arb_when_crossed(self):
        """bid_a > ask_b creates an arb opportunity."""
        det = ArbitrageDetector(threshold_bps=0.0, estimated_cost_bps=0.0)
        state = _make_state(100.0, 100.10, 100.20, 99.80, 99.90)
        # bid_a=100.10, ask_b=99.90 => spread=0.20 > 0
        opps = det.detect(state)
        arb_opps = [o for o in opps if o.kind == OpportunityKind.cross_venue_arb]
        assert len(arb_opps) > 0

    def test_arb_value_positive(self):
        det = ArbitrageDetector(threshold_bps=0.0, estimated_cost_bps=0.0)
        state = _make_state(100.0, 100.10, 100.20, 99.80, 99.90)
        opps = det.detect(state)
        for o in opps:
            assert o.estimated_value_bps > 0


class TestStaleQuoteDetector:
    def test_no_stale_when_aligned(self):
        det = StaleQuoteDetector(threshold_bps=3.0, estimated_cost_bps=1.0)
        state = _make_state(100.0, 99.95, 100.05, 99.92, 100.08)
        opps = det.detect(state)
        assert len(opps) == 0

    def test_buy_when_mid_moved_up(self):
        """True mid moved up, venue B ask is stale-low."""
        det = StaleQuoteDetector(threshold_bps=3.0, estimated_cost_bps=0.0)
        # true_mid=101.0, venue B ask still at 100.05 (stale)
        state = _make_state(101.0, 100.95, 101.05, 99.92, 100.05)
        opps = det.detect(state)
        buy_opps = [o for o in opps if o.side == SideKind.buy]
        assert len(buy_opps) > 0

    def test_sell_when_mid_moved_down(self):
        """True mid moved down, venue B bid is stale-high."""
        det = StaleQuoteDetector(threshold_bps=3.0, estimated_cost_bps=0.0)
        # true_mid=99.0, venue B bid still at 99.92 (stale)
        state = _make_state(99.0, 98.95, 99.05, 99.92, 100.08)
        opps = det.detect(state)
        sell_opps = [o for o in opps if o.side == SideKind.sell]
        assert len(sell_opps) > 0


class TestLiquidationDetector:
    def test_no_detection_without_event(self):
        det = LiquidationDetector(threshold_bps=10.0, estimated_cost_bps=0.0)
        state = _make_state(100.0, 99.50, 100.05, 99.92, 100.08, has_liq=False)
        opps = det.detect(state)
        assert len(opps) == 0

    def test_detects_with_event_and_depression(self):
        det = LiquidationDetector(threshold_bps=10.0, estimated_cost_bps=0.0)
        # true_mid=100, bid_a depressed to 99.5 => depression = 50 bps > 10
        state = _make_state(100.0, 99.50, 100.05, 99.92, 100.08, has_liq=True)
        opps = det.detect(state)
        assert len(opps) == 1
        assert opps[0].kind == OpportunityKind.liquidation

    def test_no_detection_below_threshold(self):
        det = LiquidationDetector(threshold_bps=100.0, estimated_cost_bps=0.0)
        state = _make_state(100.0, 99.50, 100.05, 99.92, 100.08, has_liq=True)
        opps = det.detect(state)
        assert len(opps) == 0
