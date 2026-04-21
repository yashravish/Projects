"""Unit tests for execution model and metrics."""

import pytest

from execsim.detectors.base import SideKind
from execsim.execution.model import FillResult, compute_metrics


class TestComputeMetrics:
    """Test metric computation against the specification.

    Definitions:
      - impl_shortfall_bps = (exec_price - arrival_mid) * signed_qty
          / (arrival_mid * |qty|) * 10000
        where signed_qty > 0 for buys, < 0 for sells.
      - realized_slippage_bps = (exec_price - decision_price)
          / decision_price * 10000
      - fill_quality = filled_qty / requested_qty
    """

    def test_buy_no_slippage(self):
        """Buy where exec_price == decision_price == arrival_mid."""
        fill = FillResult(
            venue="venue_a",
            requested_qty=1.0,
            filled_qty=1.0,
            exec_price=100.0,
            decision_price=100.0,
            arrival_mid=100.0,
            latency_steps=0,
        )
        m = compute_metrics(fill, SideKind.buy)
        assert m.impl_shortfall_bps == 0.0
        assert m.realized_slippage_bps == 0.0
        assert m.fill_quality == 1.0

    def test_buy_adverse_shortfall(self):
        """Buy at worse price than arrival mid => positive shortfall."""
        fill = FillResult(
            venue="venue_a",
            requested_qty=1.0,
            filled_qty=1.0,
            exec_price=101.0,
            decision_price=100.5,
            arrival_mid=100.0,
            latency_steps=0,
        )
        m = compute_metrics(fill, SideKind.buy)
        # IS = (101 - 100) * 1 / (100 * 1) * 10000 = 100 bps
        assert abs(m.impl_shortfall_bps - 100.0) < 1e-10
        # Slippage = (101 - 100.5) / 100.5 * 10000
        expected_slip = (101.0 - 100.5) / 100.5 * 10000.0
        assert abs(m.realized_slippage_bps - expected_slip) < 1e-10

    def test_sell_favorable_shortfall(self):
        """Sell at better price than arrival mid => negative shortfall (gain)."""
        fill = FillResult(
            venue="venue_a",
            requested_qty=1.0,
            filled_qty=1.0,
            exec_price=101.0,
            decision_price=100.5,
            arrival_mid=100.0,
            latency_steps=0,
        )
        m = compute_metrics(fill, SideKind.sell)
        # IS = (101 - 100) * (-1) / (100 * 1) * 10000 = -100 bps
        assert abs(m.impl_shortfall_bps - (-100.0)) < 1e-10

    def test_partial_fill(self):
        fill = FillResult(
            venue="venue_a",
            requested_qty=10.0,
            filled_qty=7.0,
            exec_price=100.0,
            decision_price=100.0,
            arrival_mid=100.0,
            latency_steps=0,
        )
        m = compute_metrics(fill, SideKind.buy)
        assert abs(m.fill_quality - 0.7) < 1e-10

    def test_fill_quality_capped_at_one(self):
        """fill_quality cannot exceed 1.0."""
        fill = FillResult(
            venue="venue_a",
            requested_qty=1.0,
            filled_qty=1.0,
            exec_price=100.0,
            decision_price=100.0,
            arrival_mid=100.0,
            latency_steps=0,
        )
        m = compute_metrics(fill, SideKind.buy)
        assert m.fill_quality <= 1.0
