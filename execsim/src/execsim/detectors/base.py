"""Detector protocol and shared types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from execsim.simulator.orderbook import OrderBook
from execsim.simulator.amm import AMMPool


class OpportunityKind(str, Enum):
    cross_venue_arb = "cross_venue_arb"
    stale_quote = "stale_quote"
    liquidation = "liquidation"


class SideKind(str, Enum):
    buy = "buy"
    sell = "sell"


@dataclass(frozen=True)
class DetectedOpportunity:
    """An opportunity detected by a detector at a single simulation step.

    Attributes:
        kind: Type of opportunity.
        side: Buy or sell.
        estimated_value_bps: Heuristic estimated value in basis points.
        edge_bps: Raw edge before estimated costs, in basis points.
        arrival_mid: True mid-price at detection time. Units: quote/base.
        detail: Arbitrary metadata for debugging / persistence.
    """
    kind: OpportunityKind
    side: SideKind
    estimated_value_bps: float
    edge_bps: float
    arrival_mid: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    """Complete market state at a single simulation step.

    Passed to each detector for evaluation.
    """
    step: int
    true_mid: float
    venue_a_book: OrderBook
    venue_b_book: OrderBook
    amm_pool: AMMPool
    has_liquidation: bool


class Detector(Protocol):
    """Protocol for opportunity detectors."""

    def detect(self, state: MarketState) -> list[DetectedOpportunity]:
        """Evaluate market state and return any detected opportunities.

        Args:
            state: Current market state snapshot.

        Returns:
            List of detected opportunities (may be empty).
        """
        ...
