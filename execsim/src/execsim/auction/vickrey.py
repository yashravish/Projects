"""Vickrey (sealed-bid, second-price, single-unit) auction.

Mechanism:
  For each opportunity (single-unit item):
    1. Collect all bids.
    2. Discard bids below reserve_price_bps.
    3. If no valid bids remain, the opportunity is unallocated.
    4. Winner = highest bid. Payment = max(second-highest bid, reserve_price_bps).
    5. Tie-breaking: lowest bidder index (deterministic).

Vickrey incentive property: truthful bidding is a weakly dominant strategy.
Under this mechanism, a bidder cannot improve their payoff by deviating
from their true valuation.

This module operates on in-memory data structures. Persistence is handled
by the caller.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Bid:
    """A single bid in the auction.

    Attributes:
        opportunity_index: Index of the opportunity being bid on.
        bidder_index: Index of the bidder (for tie-breaking).
        value_bps: Bid value in basis points.
    """
    opportunity_index: int
    bidder_index: int
    value_bps: float


@dataclass(frozen=True)
class AuctionOutcome:
    """Outcome for a single opportunity in the auction.

    Attributes:
        opportunity_index: Which opportunity this outcome is for.
        allocated: Whether the opportunity was allocated.
        winner_index: Bidder index of the winner (None if not allocated).
        winning_bid_bps: Winning bid value (None if not allocated).
        payment_bps: Payment under second-price rule (0 if not allocated).
        all_bids: All bids for this opportunity (for persistence).
    """
    opportunity_index: int
    allocated: bool
    winner_index: int | None
    winning_bid_bps: float | None
    payment_bps: float
    all_bids: list[Bid]


@dataclass
class AuctionResult:
    """Aggregate result of running the auction.

    Attributes:
        outcomes: Per-opportunity outcomes.
        total_revenue_bps: Sum of payments across all allocated opportunities.
        num_opportunities: Total number of opportunities.
        num_allocated: Number of allocated opportunities.
        allocation_rate: num_allocated / num_opportunities (0 if no opportunities).
        mean_payment_bps: Mean payment across allocated opportunities (0 if none).
    """
    outcomes: list[AuctionOutcome]
    total_revenue_bps: float
    num_opportunities: int
    num_allocated: int
    allocation_rate: float
    mean_payment_bps: float


def generate_synthetic_bids(
    opportunity_values_bps: list[float],
    n_bidders: int,
    rng: np.random.Generator,
) -> list[list[Bid]]:
    """Generate synthetic bids for each opportunity.

    Each opportunity gets n_bidders bids. Bid values are drawn as:
        bid = opportunity_value * U(0.7, 1.3)

    This creates a distribution of bids for testing the auction mechanism.
    The first bidder always bids the true value (truthful bidder).

    Args:
        opportunity_values_bps: Estimated value of each opportunity in bps.
        n_bidders: Number of synthetic bidders per opportunity. Must be >= 1.
        rng: Seeded numpy random generator.

    Returns:
        List of bid lists, one per opportunity.
    """
    if n_bidders < 1:
        raise ValueError(f"n_bidders must be >= 1, got {n_bidders}")

    all_bids: list[list[Bid]] = []

    for opp_idx, value in enumerate(opportunity_values_bps):
        opp_bids: list[Bid] = []

        # First bidder is always truthful
        opp_bids.append(Bid(
            opportunity_index=opp_idx,
            bidder_index=0,
            value_bps=value,
        ))

        # Additional bidders draw from U(0.7, 1.3) * value
        for bidder_idx in range(1, n_bidders):
            noise = float(rng.uniform(0.7, 1.3))
            opp_bids.append(Bid(
                opportunity_index=opp_idx,
                bidder_index=bidder_idx,
                value_bps=value * noise,
            ))

        all_bids.append(opp_bids)

    return all_bids


def run_vickrey_auction(
    bids_per_opportunity: list[list[Bid]],
    reserve_price_bps: float,
) -> AuctionResult:
    """Run a Vickrey auction over all opportunities.

    For each opportunity:
      1. Filter out bids below reserve_price_bps.
      2. If no valid bids, mark as unallocated.
      3. Winner = highest bid. Ties broken by lowest bidder_index.
      4. Payment = max(second-highest valid bid, reserve_price_bps).

    Args:
        bids_per_opportunity: List of bid lists, one per opportunity.
        reserve_price_bps: Reserve price in bps. Must be >= 0.

    Returns:
        AuctionResult with per-opportunity outcomes and aggregates.
    """
    if reserve_price_bps < 0:
        raise ValueError(f"reserve_price_bps must be >= 0, got {reserve_price_bps}")

    outcomes: list[AuctionOutcome] = []
    total_revenue = 0.0
    num_allocated = 0

    for opp_idx, bids in enumerate(bids_per_opportunity):
        # Filter bids at or above reserve
        valid = [b for b in bids if b.value_bps >= reserve_price_bps]

        if not valid:
            outcomes.append(AuctionOutcome(
                opportunity_index=opp_idx,
                allocated=False,
                winner_index=None,
                winning_bid_bps=None,
                payment_bps=0.0,
                all_bids=bids,
            ))
            continue

        # Sort by bid descending, then by bidder_index ascending (tie-break)
        valid_sorted = sorted(valid, key=lambda b: (-b.value_bps, b.bidder_index))
        winner = valid_sorted[0]

        # Payment = max(second-highest bid, reserve)
        if len(valid_sorted) >= 2:
            second_price = valid_sorted[1].value_bps
            payment = max(second_price, reserve_price_bps)
        else:
            payment = reserve_price_bps

        outcomes.append(AuctionOutcome(
            opportunity_index=opp_idx,
            allocated=True,
            winner_index=winner.bidder_index,
            winning_bid_bps=winner.value_bps,
            payment_bps=payment,
            all_bids=bids,
        ))
        total_revenue += payment
        num_allocated += 1

    num_opps = len(bids_per_opportunity)
    allocation_rate = num_allocated / num_opps if num_opps > 0 else 0.0
    mean_payment = total_revenue / num_allocated if num_allocated > 0 else 0.0

    return AuctionResult(
        outcomes=outcomes,
        total_revenue_bps=total_revenue,
        num_opportunities=num_opps,
        num_allocated=num_allocated,
        allocation_rate=allocation_rate,
        mean_payment_bps=mean_payment,
    )
