# Architecture Decision Records

## ADR-1: Sync SQLAlchemy over async

**Context**: FastAPI supports async handlers with asyncpg. The workload
here is CPU-bound simulation (numpy), not I/O-bound serving.

**Decision**: Use sync SQLAlchemy with psycopg2. FastAPI runs sync handlers
in a threadpool.

**Consequences**: Simpler code, no async session management. If the service
needed to handle many concurrent I/O-bound requests, async would be better.
That is not this use case.

## ADR-2: Per-step volatility, not annualized

**Context**: Standard GBM uses annualized volatility. With dt=0.1s and
1000 steps, annualized vol would produce near-invisible price moves.

**Decision**: `sigma` is per-step volatility (e.g., 0.02 means ~2% per step).
This is documented in config and price_process.py.

**Consequences**: Prices show meaningful movement in short simulations.
The parameter is not directly comparable to real-world annualized vol.
Clearly documented as simulated.

## ADR-3: Synthetic AMM rebalance

**Context**: A realistic AMM would change price only through arbitrageur
trades against the pool. Simulating arbitrageur behavior is complex and
out of scope.

**Decision**: Directly adjust reserves each step to track the true mid-price
with log-normal noise. k is preserved.

**Consequences**: AMM price correlates with but is not identical to true mid.
Not a realistic AMM dynamic. Labeled as a simulation simplification.

## ADR-4: Fixed 1-unit execution quantity

**Context**: Variable execution sizes add complexity in metric comparison
and normalization.

**Decision**: All opportunities are executed at 1.0 unit. Metrics are
directly comparable across opportunities.

**Consequences**: Does not model quantity-dependent effects (larger orders
have more slippage). Acceptable for a simulation focused on detection and
auction mechanics. Noted in limitations.

## ADR-5: Synthetic auction bidders

**Context**: The project needs to exercise the Vickrey mechanism, but
there are no real competing strategies.

**Decision**: Generate n synthetic bidders per opportunity. First bidder
is truthful, others draw from U(0.7, 1.3) * estimated_value. Seeded.

**Consequences**: Tests the auction mechanism and calibration loop.
Does not model strategic bidder behavior. Acceptable for demonstrating
the Vickrey incentive property on constructed inputs.

## ADR-6: Grid search for calibration

**Context**: More sophisticated methods (Bayesian optimization) are
out of scope per project requirements.

**Decision**: Exhaustive grid search over integer bps reserves. Simple,
deterministic, easy to verify.

**Consequences**: Linear time in grid size * number of held-out seeds.
With default params (51 grid points * 3 seeds), this is fast.
Not scalable to very fine grids or many seeds, but that is not needed.
