# Architecture

## Overview

execsim is a single-process Python service. It runs simulation, detection,
execution modeling, and auction logic synchronously within API request handlers.

## Data flow

```
API request (seed, config)
  -> Simulation engine
       -> GBM price process (numpy)
       -> Per-step:
            -> Build venue A book (current mid)
            -> Build venue B book (lagged mid)
            -> Rebalance AMM pool (synthetic)
            -> Roll liquidation event (Bernoulli)
            -> Run detectors (arb, stale, liquidation)
            -> Simulate fills (VWAP for LOB, constant-product for AMM)
            -> Compute metrics (IS, slippage, fill quality)
  -> Persist to PostgreSQL
  -> Return run summary

API request (run_id, reserve, n_bidders)
  -> Generate synthetic bids per opportunity
  -> Run Vickrey auction
  -> Persist auction entries and result
  -> Return auction summary

API request (seeds, grid params)
  -> For each grid point:
       -> For each held-out seed:
            -> Run simulation
            -> Run auction
       -> Compute mean revenue and allocation rate
  -> Select feasible optimum
  -> Return calibration result
```

## Key components

### Simulator

- **Price process**: GBM (Euler-Maruyama on log-price). Per-step volatility,
  not annualized. Produces strictly positive prices.
- **Order book**: Reconstructed each step from mid-price. Not persistent
  across steps. 5 levels per side, uniform random quantities.
- **AMM**: Constant-product (x*y=k) with fee on input. Reserves are
  synthetically rebalanced to track the true mid with log-normal noise.
  This is a simulation simplification.
- **Liquidation**: Bernoulli event per step. Depresses venue A bid by
  configurable bps.

### Detectors

Three detectors, each implementing the `Detector` protocol:
1. **ArbitrageDetector**: Checks all venue pairs for crossed quotes.
2. **StaleQuoteDetector**: Checks venue B vs true mid for staleness gaps.
3. **LiquidationDetector**: Checks venue A bid depression during liq events.

All value estimates are labeled as heuristics.

### Execution model

- LOB fills use VWAP across book levels.
- AMM fills use constant-product formula with fee.
- Fixed 1-unit quantity per opportunity.
- Venue-specific latency in steps.

### Metrics

Defined precisely (see `execution/model.py` docstring):
- **Implementation shortfall**: `(exec_price - arrival_mid) * signed_qty / (arrival_mid * |qty|) * 10000` bps
- **Realized slippage**: `(exec_price - decision_price) / decision_price * 10000` bps
- **Fill quality**: `filled_qty / requested_qty`, in [0, 1]

### Auction

Vickrey (sealed-bid, second-price, single-unit per opportunity). Synthetic
bidders with noise around estimated value. Tie-breaking by lowest bidder index.

### Calibration

Grid search over integer bps reserve prices. Objective: maximize mean revenue
on held-out seeds, subject to `allocation_rate >= floor`. Exhaustive evaluation.

### Validation

Rule-based checkers (not AI, not ML):
- Schema: null fields, negative prices, qty violations
- Temporal: monotone steps/timestamps, ordering constraints
- State: AMM reserves, exec price sanity
- Calibration drift: re-calibration vs stored optimal

## Persistence

PostgreSQL 16, accessed via SQLAlchemy 2.x (sync). All models use UUID PKs,
JSONB for flexible detail fields, and native PostgreSQL enums.

## Configuration

All parameters via environment variables (pydantic-settings). See `config.py`.
