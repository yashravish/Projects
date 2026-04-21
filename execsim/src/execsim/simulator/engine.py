"""Main simulation engine.

Orchestrates the simulation loop: generates prices, constructs venues,
runs detectors, simulates execution, computes metrics, and persists results.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import structlog

from execsim.config import Settings
from execsim.detectors.arbitrage import ArbitrageDetector
from execsim.detectors.base import DetectedOpportunity, MarketState, SideKind
from execsim.detectors.liquidation import LiquidationDetector
from execsim.detectors.stale_quote import StaleQuoteDetector
from execsim.execution.model import (
    ExecutionMetrics,
    FillResult,
    compute_metrics,
    simulate_fill,
)
from execsim.simulator.amm import AMMPool, create_pool, rebalance_pool
from execsim.simulator.events import apply_liquidation_impact, should_liquidate
from execsim.simulator.orderbook import OrderBook, build_order_book
from execsim.simulator.price_process import generate_mid_prices

log = structlog.get_logger()


@dataclass
class SnapshotRecord:
    """In-memory record of a single simulation step."""
    step: int
    true_mid: float
    venue_a_bid: float
    venue_a_ask: float
    venue_b_bid: float
    venue_b_ask: float
    amm_reserve_x: float
    amm_reserve_y: float
    amm_price: float
    has_liquidation: bool
    ts: datetime


@dataclass
class OpportunityRecord:
    """In-memory record of a detected opportunity."""
    id: uuid.UUID
    step: int
    kind: str
    side: str
    estimated_value_bps: float
    edge_bps: float
    arrival_mid: float
    detail: dict
    detected_at: datetime


@dataclass
class FillRecord:
    """In-memory record of a simulated fill."""
    opportunity_id: uuid.UUID
    venue: str
    requested_qty: float
    filled_qty: float
    exec_price: float
    decision_price: float
    arrival_mid: float
    latency_steps: int
    executed_at: datetime


@dataclass
class MetricRecord:
    """In-memory record of execution metrics for a fill."""
    impl_shortfall_bps: float
    realized_slippage_bps: float
    fill_quality: float


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""
    seed: int
    num_steps: int
    config: dict
    started_at: datetime
    finished_at: datetime
    snapshots: list[SnapshotRecord] = field(default_factory=list)
    opportunities: list[OpportunityRecord] = field(default_factory=list)
    fills: list[FillRecord] = field(default_factory=list)
    metrics: list[MetricRecord] = field(default_factory=list)


def run_simulation(seed: int, settings: Settings) -> SimulationResult:
    """Run a complete simulation with the given seed and settings.

    This is the main entry point for the simulation. It is deterministic:
    given the same seed and settings, it produces identical results.

    Args:
        seed: Random seed for reproducibility. Must be >= 0.
        settings: Application settings containing all simulator parameters.

    Returns:
        SimulationResult containing all snapshots, opportunities, fills, and metrics.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    rng = np.random.default_rng(seed)
    started_at = datetime.now(timezone.utc)

    num_steps = settings.sim_num_steps
    dt = settings.sim_dt

    # Generate true mid-price path
    prices = generate_mid_prices(
        num_steps=num_steps,
        initial_price=settings.sim_initial_price,
        mu=settings.sim_mu,
        sigma=settings.sim_sigma,
        dt=dt,
        rng=rng,
    )

    # Initialize AMM pool
    amm_pool = create_pool(
        initial_price=settings.sim_initial_price,
        reserve_x=settings.amm_initial_x,
        fee_bps=settings.amm_fee_bps,
    )

    # Initialize detectors
    arb_detector = ArbitrageDetector(
        threshold_bps=settings.arb_threshold_bps,
        estimated_cost_bps=settings.estimated_cost_bps,
    )
    stale_detector = StaleQuoteDetector(
        threshold_bps=settings.stale_threshold_bps,
        estimated_cost_bps=settings.estimated_cost_bps,
    )
    liq_detector = LiquidationDetector(
        threshold_bps=settings.liq_threshold_bps,
        estimated_cost_bps=settings.estimated_cost_bps,
    )

    detectors = [arb_detector, stale_detector, liq_detector]

    lag = settings.venue_b_lag_steps

    snapshots: list[SnapshotRecord] = []
    opportunities: list[OpportunityRecord] = []
    fills: list[FillRecord] = []
    metrics: list[MetricRecord] = []

    config_dict = settings.model_dump()

    for t in range(num_steps):
        true_mid = float(prices[t])

        # Venue A: uses current true mid
        book_a = build_order_book(
            mid_price=true_mid,
            half_spread_bps=settings.venue_a_half_spread_bps,
            num_levels=settings.venue_levels,
            tick_size=settings.venue_tick_size,
            min_level_qty=settings.venue_min_level_qty,
            max_level_qty=settings.venue_max_level_qty,
            rng=rng,
        )

        # Venue B: uses lagged true mid
        lagged_t = max(0, t - lag)
        lagged_mid = float(prices[lagged_t])
        book_b = build_order_book(
            mid_price=lagged_mid,
            half_spread_bps=settings.venue_b_half_spread_bps,
            num_levels=settings.venue_levels,
            tick_size=settings.venue_tick_size,
            min_level_qty=settings.venue_min_level_qty,
            max_level_qty=settings.venue_max_level_qty,
            rng=rng,
        )

        # Rebalance AMM
        amm_pool = rebalance_pool(
            pool=amm_pool,
            target_price=true_mid,
            noise_std=settings.amm_tracking_noise_std,
            rng=rng,
        )

        # Check for liquidation event
        has_liq = should_liquidate(settings.liq_probability, rng)
        if has_liq:
            depressed_bid = apply_liquidation_impact(
                best_bid=book_a.best_bid,
                impact_bps=settings.liq_impact_bps,
            )
            # Rebuild book_a with depressed bid
            from execsim.simulator.orderbook import BookLevel
            new_bids = [BookLevel(price=depressed_bid, qty=book_a.bids[0].qty)]
            new_bids.extend(
                BookLevel(
                    price=min(depressed_bid - (i + 1) * settings.venue_tick_size, b.price),
                    qty=b.qty,
                )
                for i, b in enumerate(book_a.bids[1:])
            )
            book_a = OrderBook(bids=new_bids, asks=book_a.asks)

        step_ts = datetime.now(timezone.utc)

        # Record snapshot
        snapshots.append(SnapshotRecord(
            step=t,
            true_mid=true_mid,
            venue_a_bid=book_a.best_bid,
            venue_a_ask=book_a.best_ask,
            venue_b_bid=book_b.best_bid,
            venue_b_ask=book_b.best_ask,
            amm_reserve_x=amm_pool.reserve_x,
            amm_reserve_y=amm_pool.reserve_y,
            amm_price=amm_pool.spot_price,
            has_liquidation=has_liq,
            ts=step_ts,
        ))

        # Build market state for detectors
        market_state = MarketState(
            step=t,
            true_mid=true_mid,
            venue_a_book=book_a,
            venue_b_book=book_b,
            amm_pool=amm_pool,
            has_liquidation=has_liq,
        )

        # Run detectors
        for detector in detectors:
            detected = detector.detect(market_state)
            for opp in detected:
                opp_id = uuid.uuid4()
                opportunities.append(OpportunityRecord(
                    id=opp_id,
                    step=t,
                    kind=opp.kind.value,
                    side=opp.side.value,
                    estimated_value_bps=opp.estimated_value_bps,
                    edge_bps=opp.edge_bps,
                    arrival_mid=opp.arrival_mid,
                    detail=opp.detail,
                    detected_at=step_ts,
                ))

                # Determine execution venue based on opportunity type
                exec_venue = _select_execution_venue(opp)
                exec_side = SideKind(opp.side.value)
                latency = _venue_latency(exec_venue, lag)

                # Simulate fill against current state (simplified: no look-ahead)
                fill_result = simulate_fill(
                    venue=exec_venue,
                    side=exec_side,
                    qty=1.0,  # fixed qty per plan
                    state=market_state,
                    latency_steps=latency,
                )

                if fill_result.filled_qty > 0:
                    fills.append(FillRecord(
                        opportunity_id=opp_id,
                        venue=fill_result.venue,
                        requested_qty=fill_result.requested_qty,
                        filled_qty=fill_result.filled_qty,
                        exec_price=fill_result.exec_price,
                        decision_price=fill_result.decision_price,
                        arrival_mid=fill_result.arrival_mid,
                        latency_steps=fill_result.latency_steps,
                        executed_at=step_ts,
                    ))

                    metric = compute_metrics(fill_result, exec_side)
                    metrics.append(MetricRecord(
                        impl_shortfall_bps=metric.impl_shortfall_bps,
                        realized_slippage_bps=metric.realized_slippage_bps,
                        fill_quality=metric.fill_quality,
                    ))

    finished_at = datetime.now(timezone.utc)

    log.info(
        "simulation_complete",
        seed=seed,
        num_steps=num_steps,
        num_opportunities=len(opportunities),
        num_fills=len(fills),
    )

    return SimulationResult(
        seed=seed,
        num_steps=num_steps,
        config=config_dict,
        started_at=started_at,
        finished_at=finished_at,
        snapshots=snapshots,
        opportunities=opportunities,
        fills=fills,
        metrics=metrics,
    )


def _select_execution_venue(opp: DetectedOpportunity) -> str:
    """Select which venue to execute on for a given opportunity.

    Args:
        opp: Detected opportunity with kind and detail.

    Returns:
        Venue name string: "venue_a", "venue_b", or "amm".
    """
    if opp.kind.value == "cross_venue_arb":
        return opp.detail.get("buy_venue", "venue_a")
    elif opp.kind.value == "stale_quote":
        return "venue_b"
    elif opp.kind.value == "liquidation":
        return "venue_a"
    else:
        return "venue_a"


def _venue_latency(venue: str, lag_steps: int) -> int:
    """Return execution latency in steps for a venue.

    Args:
        venue: Venue name.
        lag_steps: Configured lag for venue B.

    Returns:
        Latency in simulation steps.
    """
    if venue == "venue_a":
        return 0
    elif venue == "venue_b":
        return lag_steps
    elif venue == "amm":
        return 1
    return 0
