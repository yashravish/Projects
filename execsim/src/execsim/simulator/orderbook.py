"""Simulated limit order book (LOB) for a single venue.

This is a simplified order book with configurable depth, spread, and tick size.
It is reconstructed each step from the mid-price — it does not maintain persistent
order state across steps. This is a simulation simplification, not a realistic
microstructure model.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BookLevel:
    """A single price level in the order book.

    Attributes:
        price: Price at this level. Units: quote currency per base unit.
        qty: Available quantity at this level. Units: base units.
    """
    price: float
    qty: float


@dataclass(frozen=True)
class OrderBook:
    """Snapshot of one side of a limit order book.

    bids: sorted descending by price (best bid first).
    asks: sorted ascending by price (best ask first).
    """
    bids: list[BookLevel]
    asks: list[BookLevel]

    @property
    def best_bid(self) -> float:
        """Best (highest) bid price."""
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        """Best (lowest) ask price."""
        return self.asks[0].price

    @property
    def mid(self) -> float:
        """Mid-price, arithmetic mean of best bid and best ask.

        Units: quote currency per base unit.
        """
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute price terms.

        Units: quote currency.
        """
        return self.best_ask - self.best_bid

    def vwap_buy(self, qty: float) -> tuple[float, float]:
        """Compute volume-weighted average price for a market buy order.

        Walks the ask side of the book up to requested quantity.

        Args:
            qty: Requested buy quantity. Units: base units. Must be > 0.

        Returns:
            Tuple of (vwap, filled_qty).
            - vwap: volume-weighted average price. Units: quote/base.
              Returns 0.0 if filled_qty is 0.
            - filled_qty: actual quantity filled (<= qty). Units: base units.
        """
        return _walk_levels(self.asks, qty)

    def vwap_sell(self, qty: float) -> tuple[float, float]:
        """Compute volume-weighted average price for a market sell order.

        Walks the bid side of the book down to requested quantity.

        Args:
            qty: Requested sell quantity. Units: base units. Must be > 0.

        Returns:
            Tuple of (vwap, filled_qty). Same semantics as vwap_buy.
        """
        return _walk_levels(self.bids, qty)


def _walk_levels(levels: list[BookLevel], qty: float) -> tuple[float, float]:
    """Walk price levels to fill a quantity.

    Args:
        levels: Ordered price levels (best first).
        qty: Requested quantity (base units), must be > 0.

    Returns:
        (vwap, filled_qty). vwap is 0.0 if filled_qty is 0.
    """
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")

    remaining = qty
    total_cost = 0.0
    filled = 0.0

    for level in levels:
        take = min(remaining, level.qty)
        total_cost += take * level.price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break

    if filled <= 0:
        return 0.0, 0.0
    return total_cost / filled, filled


def build_order_book(
    mid_price: float,
    half_spread_bps: float,
    num_levels: int,
    tick_size: float,
    min_level_qty: float,
    max_level_qty: float,
    rng: np.random.Generator,
) -> OrderBook:
    """Construct a synthetic order book around a mid-price.

    Args:
        mid_price: Center price for the book. Units: quote/base. Must be > 0.
        half_spread_bps: Half the bid-ask spread in basis points (1 bp = 0.01%).
        num_levels: Number of price levels per side. Must be >= 1.
        tick_size: Price increment between levels. Units: quote currency.
        min_level_qty: Minimum quantity per level. Units: base units.
        max_level_qty: Maximum quantity per level. Units: base units.
        rng: Seeded numpy random generator.

    Returns:
        An OrderBook with `num_levels` bids (descending) and asks (ascending).

    The best bid/ask are placed at mid_price * (1 -/+ half_spread_bps/10000).
    Subsequent levels are spaced by tick_size away from the mid.
    Quantities at each level are drawn from Uniform(min_level_qty, max_level_qty).
    """
    if mid_price <= 0:
        raise ValueError(f"mid_price must be > 0, got {mid_price}")

    half_spread = mid_price * half_spread_bps / 10000.0
    best_bid = mid_price - half_spread
    best_ask = mid_price + half_spread

    bid_qtys = rng.uniform(min_level_qty, max_level_qty, size=num_levels)
    ask_qtys = rng.uniform(min_level_qty, max_level_qty, size=num_levels)

    bids = [
        BookLevel(
            price=round(best_bid - i * tick_size, 8),
            qty=round(float(bid_qtys[i]), 8),
        )
        for i in range(num_levels)
    ]

    asks = [
        BookLevel(
            price=round(best_ask + i * tick_size, 8),
            qty=round(float(ask_qtys[i]), 8),
        )
        for i in range(num_levels)
    ]

    return OrderBook(bids=bids, asks=asks)
