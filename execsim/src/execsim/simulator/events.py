"""Liquidation event injection for the simulator.

Liquidation events are synthetic: they represent a configurable "distressed sell
order" that depresses the best bid on venue A for the step in which it occurs.
They are not connected to any real liquidation mechanism.
"""

import numpy as np


def should_liquidate(probability: float, rng: np.random.Generator) -> bool:
    """Decide whether a liquidation event fires this step.

    Args:
        probability: Probability of liquidation per step, in [0, 1].
        rng: Seeded numpy random generator.

    Returns:
        True if the event fires.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    return bool(rng.random() < probability)


def apply_liquidation_impact(
    best_bid: float,
    impact_bps: float,
) -> float:
    """Compute the depressed best bid after a liquidation event.

    Args:
        best_bid: Current best bid price. Units: quote/base. Must be > 0.
        impact_bps: Downward impact in basis points. Must be > 0.

    Returns:
        Depressed bid price. Units: quote/base. Always > 0 (impact is capped
        at 99% of best_bid).

    Formula:
        depressed_bid = best_bid * (1 - impact_bps / 10000)
        Capped so result > 0.
    """
    if best_bid <= 0:
        raise ValueError(f"best_bid must be > 0, got {best_bid}")
    if impact_bps < 0:
        raise ValueError(f"impact_bps must be >= 0, got {impact_bps}")

    factor = max(1.0 - impact_bps / 10000.0, 0.01)
    return best_bid * factor
