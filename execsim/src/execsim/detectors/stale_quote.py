"""Stale-quote capture detector.

Detects opportunities where venue B's quote is stale (lagging the true mid)
and the gap exceeds a configurable threshold.

Predicate:
    stale_buy  := true_mid(t) - ask_B(t) > threshold_bps * true_mid(t) / 10000
    stale_sell := bid_B(t) - true_mid(t) > threshold_bps * true_mid(t) / 10000

Value estimation (heuristic):
    estimated_value_bps = |venue_b_quote - true_mid| / true_mid * 10000 - estimated_cost_bps

This is labeled as adverse-selection capture against a lagging venue.
Fill probability and queue position are simulated, not modeled realistically.
"""

from execsim.detectors.base import (
    DetectedOpportunity,
    MarketState,
    OpportunityKind,
    SideKind,
)


class StaleQuoteDetector:
    """Detect stale-quote capture opportunities on venue B."""

    def __init__(self, threshold_bps: float, estimated_cost_bps: float):
        """
        Args:
            threshold_bps: Minimum staleness gap in bps to flag.
            estimated_cost_bps: Assumed execution cost in bps (heuristic).
        """
        if threshold_bps < 0:
            raise ValueError(f"threshold_bps must be >= 0, got {threshold_bps}")
        if estimated_cost_bps < 0:
            raise ValueError(f"estimated_cost_bps must be >= 0, got {estimated_cost_bps}")
        self.threshold_bps = threshold_bps
        self.estimated_cost_bps = estimated_cost_bps

    def detect(self, state: MarketState) -> list[DetectedOpportunity]:
        """Check venue B for stale quotes relative to the true mid.

        Args:
            state: Current market state.

        Returns:
            List of detected stale-quote opportunities (at most one per step).
        """
        mid = state.true_mid
        ask_b = state.venue_b_book.best_ask
        bid_b = state.venue_b_book.best_bid
        threshold_abs = self.threshold_bps * mid / 10000.0

        opportunities: list[DetectedOpportunity] = []

        # Buy opportunity: true mid moved up, venue B ask is stale-low
        buy_gap = mid - ask_b
        if buy_gap > threshold_abs:
            edge_bps = buy_gap / mid * 10000.0
            value_bps = edge_bps - self.estimated_cost_bps
            if value_bps > 0:
                opportunities.append(
                    DetectedOpportunity(
                        kind=OpportunityKind.stale_quote,
                        side=SideKind.buy,
                        estimated_value_bps=value_bps,
                        edge_bps=edge_bps,
                        arrival_mid=mid,
                        detail={
                            "venue": "venue_b",
                            "stale_ask": ask_b,
                            "true_mid": mid,
                            "gap": buy_gap,
                        },
                    )
                )

        # Sell opportunity: true mid moved down, venue B bid is stale-high
        sell_gap = bid_b - mid
        if sell_gap > threshold_abs:
            edge_bps = sell_gap / mid * 10000.0
            value_bps = edge_bps - self.estimated_cost_bps
            if value_bps > 0:
                opportunities.append(
                    DetectedOpportunity(
                        kind=OpportunityKind.stale_quote,
                        side=SideKind.sell,
                        estimated_value_bps=value_bps,
                        edge_bps=edge_bps,
                        arrival_mid=mid,
                        detail={
                            "venue": "venue_b",
                            "stale_bid": bid_b,
                            "true_mid": mid,
                            "gap": sell_gap,
                        },
                    )
                )

        return opportunities
