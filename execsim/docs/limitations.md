# Limitations

What this project does not do, and why.

## Not a real trading system

This is a simulator. All market data is generated, not sourced from a real
exchange or blockchain. No real orders are placed. No real assets are
traded. Performance numbers (fill rates, slippage) are simulated, not
measured against a live market.

## No real market data ingestion

The simulator generates synthetic prices via GBM. It does not connect to
any data feed, exchange API, or blockchain node.

## No authentication or multi-tenancy

The API is unauthenticated. All data is shared. This is a single-user
development tool.

## No frontend

API only. No web UI, no charts, no dashboards.

## No distributed execution

Single-process, single-threaded simulation. No Kubernetes, no task queues,
no distributed computing.

## Simplified order book

Books are reconstructed each step from the mid-price. There is no persistent
order state, no queue position modeling, no cancellation. Fill simulation
uses VWAP across 5 levels, not a realistic matching engine.

## Simplified AMM

The AMM pool is synthetically rebalanced to track the true mid-price each
step. In a real AMM, price changes come only from trades. The synthetic
rebalance is a simulation simplification that produces correlated but not
identical prices.

## Fixed execution quantity

All opportunities are executed at 1.0 unit. There is no modeling of
quantity-dependent slippage or position sizing.

## Synthetic auction bidders

Bidders are not strategic agents. They are drawn from a noise distribution
around the estimated value. The Vickrey incentive property is verified on
constructed inputs, not on strategic equilibria.

## No Bayesian optimization

Reserve-price calibration uses grid search. More sophisticated methods
were explicitly placed out of scope.

## No predictive accuracy claims

The detectors use heuristic value estimates. No claim is made about their
predictive accuracy in a real market. The execution model does not account
for adverse selection, queue priority, or market impact beyond the
simplified VWAP and constant-product formulas.

## Latency is simulated

Venue latency is modeled as an integer number of simulation steps, not
real-world nanoseconds or milliseconds. No hardware latency, network
jitter, or co-location effects are modeled.

## Single asset pair

The simulator supports one synthetic BASE/QUOTE pair. Multi-asset
simulation is out of scope.
