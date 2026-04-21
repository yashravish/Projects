"""API endpoints for auctions and calibration."""

import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from execsim.auction.calibration import calibrate_reserve
from execsim.auction.vickrey import generate_synthetic_bids, run_vickrey_auction
from execsim.config import Settings
from execsim.db.models import (
    Auction,
    AuctionEntry,
    AuctionResult,
    Opportunity,
    RunStatus,
    SimulationRun,
)
from execsim.dependencies import get_db, get_settings
from execsim.schemas.auctions import (
    AuctionCreate,
    AuctionDetail,
    AuctionEntrySchema,
    AuctionResponse,
    AuctionResultSchema,
    CalibrationGridPoint,
    CalibrationRequest,
    CalibrationResponse,
)
from execsim.schemas.common import ErrorResponse

router = APIRouter(tags=["auctions"])


@router.post(
    "/runs/{run_id}/auction",
    response_model=AuctionResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    summary="Run auction on a completed run",
)
def create_auction(
    run_id: uuid.UUID,
    body: AuctionCreate,
    db: Session = Depends(get_db),
) -> AuctionResponse:
    """Run a Vickrey auction over all opportunities in a completed run."""
    run = db.query(SimulationRun).filter(SimulationRun.id == run_id).first()
    if run is None:
        raise HTTPException(404, detail=f"Run {run_id} not found")
    if run.status != RunStatus.completed:
        raise HTTPException(409, detail=f"Run {run_id} is not completed (status={run.status.value})")

    opportunities = (
        db.query(Opportunity)
        .filter(Opportunity.run_id == run_id)
        .order_by(Opportunity.step)
        .all()
    )

    opp_values = [o.estimated_value_bps for o in opportunities]
    rng = np.random.default_rng(run.seed)
    bids = generate_synthetic_bids(opp_values, body.n_bidders, rng)
    result = run_vickrey_auction(bids, body.reserve_price_bps)

    # Persist auction
    auction = Auction(
        id=uuid.uuid4(),
        run_id=run_id,
        reserve_price_bps=body.reserve_price_bps,
        num_opportunities=result.num_opportunities,
        num_allocated=result.num_allocated,
        created_at=datetime.now(timezone.utc),
    )
    db.add(auction)

    for outcome in result.outcomes:
        opp = opportunities[outcome.opportunity_index]
        for bid in outcome.all_bids:
            is_winner = outcome.allocated and bid.bidder_index == outcome.winner_index
            db.add(AuctionEntry(
                id=uuid.uuid4(),
                auction_id=auction.id,
                opportunity_id=opp.id,
                bidder_index=bid.bidder_index,
                bid_value_bps=bid.value_bps,
                won=is_winner,
                payment_bps=outcome.payment_bps if is_winner else 0.0,
            ))

    db.add(AuctionResult(
        id=uuid.uuid4(),
        auction_id=auction.id,
        total_revenue_bps=result.total_revenue_bps,
        allocation_rate=result.allocation_rate,
        mean_payment_bps=result.mean_payment_bps,
    ))

    db.commit()

    return AuctionResponse(
        id=auction.id,
        run_id=run_id,
        reserve_price_bps=auction.reserve_price_bps,
        num_opportunities=auction.num_opportunities,
        num_allocated=auction.num_allocated,
        created_at=auction.created_at,
    )


@router.get(
    "/auctions/{auction_id}",
    response_model=AuctionDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Get auction detail",
)
def get_auction(
    auction_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AuctionDetail:
    """Get full auction detail including entries and aggregate result."""
    auction = db.query(Auction).filter(Auction.id == auction_id).first()
    if auction is None:
        raise HTTPException(404, detail=f"Auction {auction_id} not found")

    entries = db.query(AuctionEntry).filter(AuctionEntry.auction_id == auction_id).all()
    result = db.query(AuctionResult).filter(AuctionResult.auction_id == auction_id).first()

    return AuctionDetail(
        id=auction.id,
        run_id=auction.run_id,
        reserve_price_bps=auction.reserve_price_bps,
        num_opportunities=auction.num_opportunities,
        num_allocated=auction.num_allocated,
        created_at=auction.created_at,
        entries=[
            AuctionEntrySchema(
                id=e.id,
                opportunity_id=e.opportunity_id,
                bidder_index=e.bidder_index,
                bid_value_bps=e.bid_value_bps,
                won=e.won,
                payment_bps=e.payment_bps,
            )
            for e in entries
        ],
        result=AuctionResultSchema(
            total_revenue_bps=result.total_revenue_bps,
            allocation_rate=result.allocation_rate,
            mean_payment_bps=result.mean_payment_bps,
        ) if result else None,
    )


@router.post(
    "/calibrate",
    response_model=CalibrationResponse,
    summary="Run reserve-price calibration",
)
def run_calibration(
    body: CalibrationRequest,
    settings: Settings = Depends(get_settings),
) -> CalibrationResponse:
    """Run grid search to find optimal reserve price.

    Objective: maximize expected auctioneer revenue on held-out seeds,
    subject to allocation_rate >= allocation_floor.
    """
    result = calibrate_reserve(
        held_out_seeds=body.held_out_seeds,
        n_bidders=body.n_bidders,
        grid_max_bps=body.grid_max_bps,
        grid_step_bps=body.grid_step_bps,
        allocation_floor=body.allocation_floor,
        settings=settings,
    )

    return CalibrationResponse(
        optimal_reserve_bps=result.optimal_reserve_bps,
        optimal_revenue_bps=result.optimal_revenue_bps,
        optimal_allocation_rate=result.optimal_allocation_rate,
        grid=[
            CalibrationGridPoint(
                reserve_bps=gp.reserve_bps,
                mean_revenue_bps=gp.mean_revenue_bps,
                mean_allocation_rate=gp.mean_allocation_rate,
                feasible=gp.feasible,
            )
            for gp in result.grid
        ],
    )
