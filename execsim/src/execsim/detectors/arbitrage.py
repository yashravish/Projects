"""Cross-venue arbitrage detector.

Detects opportunities where the best bid on one venue exceeds the best ask
on another venue by more than a configurable threshold (after estimated costs).

Predicate:
    cross_arb_exists := (bid_X - ask_Y) > threshold_bps * ask_Y / 10000
    for any pair (X, Y) in {venue_a, venue_b, amm}

Value estimation (heuristic):
    estimated_value_bps = (bid_X - ask_Y) / arrival_mid * 10000 - estimated_cost_bps

This is a heuristic — it assumes full fill at top-of-book on both legs.
"""

from execsim.detectors.base import (
    DetectedOpportunity,
    MarketState,
    OpportunityKind,
    SideKind,
)


class ArbitrageDetector:
    """Detect cross-venue arbitrage between CEX-CEX and CEX-AMM pairs."""

    def __init__(self, threshold_bps: float, estimated_cost_bps: float):
        """
        Args:
            threshold_bps: Minimum spread in bps to flag as opportunity.
            estimated_cost_bps: Assumed execution cost in bps (heuristic).
        """
        if threshold_bps < 0:
            raise ValueError(f"threshold_bps must be >= 0, got {threshold_bps}")
        if estimated_cost_bps < 0:
            raise ValueError(f"estimated_cost_bps must be >= 0, got {estimated_cost_bps}")
        self.threshold_bps = threshold_bps
        self.estimated_cost_bps = estimated_cost_bps

    def detect(self, state: MarketState) -> list[DetectedOpportunity]:
        """Check all venue pairs for cross-venue arbitrage.

        Args:
            state: Current market state with order books and AMM pool.

        Returns:
            List of detected arbitrage opportunities.
        """
        opportunities: list[DetectedOpportunity] = []
        mid = state.true_mid

        # Build list of (name, bid, ask)
        venues = [
            ("venue_a", state.venue_a_book.best_bid, state.venue_a_book.best_ask),
            ("venue_b", state.venue_b_book.best_bid, state.venue_b_book.best_ask),
            ("amm", state.amm_pool.spot_price, state.amm_pool.spot_price),
        ]

        for i, (name_x, bid_x, _ask_x) in enumerate(venues):
            for j, (name_y, _bid_y, ask_y) in enumerate(venues):
                if i == j:
                    continue

                spread = bid_x - ask_y
                threshold_abs = self.threshold_bps * ask_y / 10000.0

                if spread > threshold_abs:
                    edge_bps = spread / mid * 10000.0
                    value_bps = edge_bps - self.estimated_cost_bps

                    if value_bps > 0:
                        opportunities.append(
                            DetectedOpportunity(
                                kind=OpportunityKind.cross_venue_arb,
                                side=SideKind.buy,
                                estimated_value_bps=value_bps,
                                edge_bps=edge_bps,
                                arrival_mid=mid,
                                detail={
                                    "buy_venue": name_y,
                                    "sell_venue": name_x,
                                    "buy_price": ask_y,
                                    "sell_price": bid_x,
                                    "spread": spread,
                                },
                            )
                        )

        return opportunities
