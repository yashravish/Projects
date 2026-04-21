"""Reserve-price calibration via grid search.

Objective: maximize expected auctioneer revenue on a held-out seed set,
subject to allocation_rate >= configurable floor.

Method:
  1. For each candidate reserve in the grid [0, step, 2*step, ..., max]:
     a. For each held-out seed, run the simulation and auction.
     b. Compute mean revenue and mean allocation rate.
     c. If allocation rate < floor, mark as infeasible.
  2. Among feasible candidates, select the one with highest mean revenue.

This is a grid search, not an optimization algorithm. It evaluates every
candidate in the grid exhaustively.
"""

from dataclasses import dataclass

import numpy as np
import structlog

from execsim.auction.vickrey import generate_synthetic_bids, run_vickrey_auction
from execsim.config import Settings
from execsim.simulator.engine import run_simulation

log = structlog.get_logger()


@dataclass(frozen=True)
class GridPoint:
    """Result for a single reserve-price candidate.

    Attributes:
        reserve_bps: The candidate reserve price in basis points.
        mean_revenue_bps: Mean total revenue across held-out seeds.
        mean_allocation_rate: Mean allocation rate across held-out seeds.
        feasible: Whether mean_allocation_rate >= floor.
    """
    reserve_bps: float
    mean_revenue_bps: float
    mean_allocation_rate: float
    feasible: bool


@dataclass(frozen=True)
class CalibrationResult:
    """Full result of the calibration grid search.

    Attributes:
        optimal_reserve_bps: Reserve price that maximizes revenue among feasible candidates.
        optimal_revenue_bps: Revenue at the optimal reserve.
        optimal_allocation_rate: Allocation rate at the optimal reserve.
        grid: All evaluated grid points.
    """
    optimal_reserve_bps: float
    optimal_revenue_bps: float
    optimal_allocation_rate: float
    grid: list[GridPoint]


def calibrate_reserve(
    held_out_seeds: list[int],
    n_bidders: int,
    grid_max_bps: int,
    grid_step_bps: int,
    allocation_floor: float,
    settings: Settings,
) -> CalibrationResult:
    """Run grid search to find optimal reserve price.

    Args:
        held_out_seeds: Seeds for held-out simulation runs. Each seed produces
            one simulation and auction. Must be non-empty.
        n_bidders: Number of synthetic bidders per opportunity.
        grid_max_bps: Maximum reserve price to test (inclusive).
        grid_step_bps: Step size for the grid.
        allocation_floor: Minimum acceptable allocation rate. In [0, 1].
        settings: Application settings for the simulator.

    Returns:
        CalibrationResult with optimal reserve and full grid.

    The objective is to maximize expected auctioneer revenue on held-out seeds,
    subject to allocation_rate >= allocation_floor.
    """
    if not held_out_seeds:
        raise ValueError("held_out_seeds must be non-empty")
    if allocation_floor < 0 or allocation_floor > 1:
        raise ValueError(f"allocation_floor must be in [0, 1], got {allocation_floor}")

    # Pre-run simulations on held-out seeds
    sim_results = []
    for seed in held_out_seeds:
        result = run_simulation(seed=seed, settings=settings)
        sim_results.append(result)
        log.info("calibration_sim_complete", seed=seed, n_opps=len(result.opportunities))

    # Build grid
    reserves = list(range(0, grid_max_bps + 1, grid_step_bps))
    grid_points: list[GridPoint] = []

    for reserve in reserves:
        revenues: list[float] = []
        alloc_rates: list[float] = []

        for sim_idx, sim_result in enumerate(sim_results):
            opp_values = [o.estimated_value_bps for o in sim_result.opportunities]

            if not opp_values:
                revenues.append(0.0)
                alloc_rates.append(1.0)  # vacuously allocated
                continue

            # Use a deterministic sub-seed for bid generation
            bid_rng = np.random.default_rng(held_out_seeds[sim_idx] + reserve)
            bids = generate_synthetic_bids(opp_values, n_bidders, bid_rng)
            auction_result = run_vickrey_auction(bids, float(reserve))

            revenues.append(auction_result.total_revenue_bps)
            alloc_rates.append(auction_result.allocation_rate)

        mean_rev = sum(revenues) / len(revenues)
        mean_alloc = sum(alloc_rates) / len(alloc_rates)
        feasible = mean_alloc >= allocation_floor

        grid_points.append(GridPoint(
            reserve_bps=float(reserve),
            mean_revenue_bps=mean_rev,
            mean_allocation_rate=mean_alloc,
            feasible=feasible,
        ))

    # Select optimal among feasible
    feasible_points = [gp for gp in grid_points if gp.feasible]

    if not feasible_points:
        # No feasible point — fall back to reserve=0
        log.warning("calibration_no_feasible_point", allocation_floor=allocation_floor)
        return CalibrationResult(
            optimal_reserve_bps=0.0,
            optimal_revenue_bps=grid_points[0].mean_revenue_bps if grid_points else 0.0,
            optimal_allocation_rate=grid_points[0].mean_allocation_rate if grid_points else 0.0,
            grid=grid_points,
        )

    best = max(feasible_points, key=lambda gp: gp.mean_revenue_bps)

    log.info(
        "calibration_complete",
        optimal_reserve_bps=best.reserve_bps,
        optimal_revenue_bps=best.mean_revenue_bps,
        optimal_allocation_rate=best.mean_allocation_rate,
        grid_size=len(grid_points),
        feasible_count=len(feasible_points),
    )

    return CalibrationResult(
        optimal_reserve_bps=best.reserve_bps,
        optimal_revenue_bps=best.mean_revenue_bps,
        optimal_allocation_rate=best.mean_allocation_rate,
        grid=grid_points,
    )
