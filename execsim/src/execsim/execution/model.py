"""Execution model: fill simulation and execution metrics.

Simulates order execution against the simulated market state with:
  - Venue-specific latency (delay in steps before execution).
  - VWAP-based fill pricing for LOB venues.
  - Constant-product pricing for AMM fills.

Metric definitions (used consistently in code and docs):
  - impl_shortfall_bps = (exec_price - arrival_mid) * signed_qty
      / (arrival_mid * |qty|) * 10000
    where signed_qty > 0 for buys, < 0 for sells.
  - realized_slippage_bps = (exec_price - decision_price)
      / decision_price * 10000
  - fill_quality = filled_qty / requested_qty, in [0, 1].
"""

from dataclasses import dataclass

from execsim.detectors.base import MarketState, SideKind
from execsim.simulator.orderbook import OrderBook
from execsim.simulator.amm import AMMPool


@dataclass(frozen=True)
class FillResult:
    """Result of simulating execution of a detected opportunity.

    Attributes:
        venue: Which venue was used ("venue_a", "venue_b", or "amm").
        requested_qty: Quantity requested. Units: base units.
        filled_qty: Quantity actually filled. Units: base units.
        exec_price: Volume-weighted average execution price. Units: quote/base.
        decision_price: Best available price at detection time. Units: quote/base.
        arrival_mid: True mid-price at opportunity detection. Units: quote/base.
        latency_steps: Number of steps delay before execution.
    """
    venue: str
    requested_qty: float
    filled_qty: float
    exec_price: float
    decision_price: float
    arrival_mid: float
    latency_steps: int


@dataclass(frozen=True)
class ExecutionMetrics:
    """Computed execution quality metrics for a single fill.

    All metrics are in basis points unless otherwise noted.
    """
    impl_shortfall_bps: float
    realized_slippage_bps: float
    fill_quality: float


def simulate_fill(
    venue: str,
    side: SideKind,
    qty: float,
    state: MarketState,
    latency_steps: int,
) -> FillResult:
    """Simulate filling an order against the current market state.

    Args:
        venue: "venue_a", "venue_b", or "amm".
        side: Buy or sell.
        qty: Requested quantity in base units. Must be > 0.
        state: Market state snapshot at the execution time (i.e., after latency).
        latency_steps: Latency from detection to execution, in steps.

    Returns:
        FillResult with execution details.

    For LOB venues, execution uses VWAP across book levels.
    For AMM, execution uses the constant-product formula.
    """
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")

    arrival_mid = state.true_mid

    if venue == "amm":
        return _fill_amm(side, qty, state.amm_pool, arrival_mid, latency_steps)
    elif venue == "venue_a":
        return _fill_lob(venue, side, qty, state.venue_a_book, arrival_mid, latency_steps)
    elif venue == "venue_b":
        return _fill_lob(venue, side, qty, state.venue_b_book, arrival_mid, latency_steps)
    else:
        raise ValueError(f"Unknown venue: {venue}")


def _fill_lob(
    venue: str,
    side: SideKind,
    qty: float,
    book: OrderBook,
    arrival_mid: float,
    latency_steps: int,
) -> FillResult:
    """Fill against a limit order book using VWAP.

    Args:
        venue: Venue name for tagging.
        side: Buy or sell.
        qty: Requested quantity.
        book: Order book snapshot at execution time.
        arrival_mid: True mid at detection time.
        latency_steps: Steps of latency.

    Returns:
        FillResult. If book has insufficient depth, filled_qty < qty.
    """
    if side == SideKind.buy:
        decision_price = book.best_ask
        exec_price, filled_qty = book.vwap_buy(qty)
    else:
        decision_price = book.best_bid
        exec_price, filled_qty = book.vwap_sell(qty)

    return FillResult(
        venue=venue,
        requested_qty=qty,
        filled_qty=filled_qty,
        exec_price=exec_price,
        decision_price=decision_price,
        arrival_mid=arrival_mid,
        latency_steps=latency_steps,
    )


def _fill_amm(
    side: SideKind,
    qty: float,
    pool: AMMPool,
    arrival_mid: float,
    latency_steps: int,
) -> FillResult:
    """Fill against the AMM using constant-product pricing.

    AMM always fills 100% up to pool depth, but at the actual execution price
    from the constant-product formula.

    Args:
        side: Buy or sell.
        qty: Requested quantity.
        pool: AMM pool state at execution time.
        arrival_mid: True mid at detection time.
        latency_steps: Steps of latency.

    Returns:
        FillResult. Fills fully if pool has sufficient reserves.
    """
    decision_price = pool.spot_price

    if side == SideKind.buy:
        # Buying base: check pool has enough reserve_x
        effective_qty = min(qty, pool.reserve_x * 0.99)  # cap at 99% of reserves
        if effective_qty <= 0:
            return FillResult(
                venue="amm",
                requested_qty=qty,
                filled_qty=0.0,
                exec_price=decision_price,
                decision_price=decision_price,
                arrival_mid=arrival_mid,
                latency_steps=latency_steps,
            )
        exec_price = pool.exec_price_buy(effective_qty)
    else:
        # Selling base: always possible (adds to reserves)
        effective_qty = qty
        exec_price = pool.exec_price_sell(effective_qty)

    return FillResult(
        venue="amm",
        requested_qty=qty,
        filled_qty=effective_qty,
        exec_price=exec_price,
        decision_price=decision_price,
        arrival_mid=arrival_mid,
        latency_steps=latency_steps,
    )


def compute_metrics(fill: FillResult, side: SideKind) -> ExecutionMetrics:
    """Compute execution quality metrics for a fill.

    Args:
        fill: Completed fill result.
        side: Buy or sell (needed for signed shortfall).

    Returns:
        ExecutionMetrics with all three metrics computed.

    Formulas:
        impl_shortfall_bps = (exec_price - arrival_mid) * signed_qty
            / (arrival_mid * |qty|) * 10000
          where signed_qty = filled_qty if buy, -filled_qty if sell.

        realized_slippage_bps = (exec_price - decision_price)
            / decision_price * 10000

        fill_quality = filled_qty / requested_qty
    """
    if fill.requested_qty <= 0:
        raise ValueError("requested_qty must be > 0")
    if fill.arrival_mid <= 0:
        raise ValueError("arrival_mid must be > 0")
    if fill.decision_price <= 0:
        raise ValueError("decision_price must be > 0")

    signed_qty = fill.filled_qty if side == SideKind.buy else -fill.filled_qty

    impl_shortfall_bps = (
        (fill.exec_price - fill.arrival_mid) * signed_qty
        / (fill.arrival_mid * abs(fill.filled_qty))
        * 10000.0
    )

    realized_slippage_bps = (
        (fill.exec_price - fill.decision_price) / fill.decision_price * 10000.0
    )

    fill_quality = fill.filled_qty / fill.requested_qty

    return ExecutionMetrics(
        impl_shortfall_bps=impl_shortfall_bps,
        realized_slippage_bps=realized_slippage_bps,
        fill_quality=min(fill_quality, 1.0),
    )
