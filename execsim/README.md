# execsim

Execution Simulator & Auction Optimizer.

A self-contained Python service that simulates a multi-venue market, detects
trading opportunities, models execution quality, and runs Vickrey auctions
with reserve-price calibration. All data is simulated — this is not a real
trading system.

## Quick start

```bash
docker compose up --build -d
```

This starts PostgreSQL 16 and the API. Migrations run automatically.
API available at `http://localhost:8000/docs`.

## Running tests

```bash
# Unit and integration tests (no DB required)
python -m pytest tests/unit/ tests/integration/ -v

# Full suite including API tests (requires DB)
docker compose exec api pytest -v
```

## API

All endpoints under `/api/v1`. See `/docs` for interactive docs.

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Liveness check |
| GET | /ready | Readiness check (DB) |
| POST | /runs | Start simulation |
| GET | /runs | List runs |
| GET | /runs/{id} | Run detail |
| GET | /runs/{id}/opportunities | List opportunities |
| GET | /runs/{id}/fills | List fills |
| GET | /runs/{id}/metrics | Aggregate metrics |
| POST | /runs/{id}/auction | Run Vickrey auction |
| GET | /auctions/{id} | Auction detail |
| POST | /calibrate | Reserve-price calibration |
| POST | /runs/{id}/validate | Run validation checks |
| GET | /alerts | List validation alerts |

## Project structure

```
src/execsim/
  config.py          - pydantic-settings configuration
  main.py            - FastAPI app factory
  db/models.py       - SQLAlchemy ORM models
  schemas/           - Pydantic request/response schemas
  simulator/         - GBM price process, LOB, AMM, simulation engine
  detectors/         - Arbitrage, stale-quote, liquidation detectors
  execution/         - Fill simulation and metric computation
  auction/           - Vickrey auction and reserve calibration
  validation/        - Schema, temporal, state, calibration checkers
  api/               - FastAPI route handlers
```

See `docs/architecture.md` for design details, `docs/decisions.md` for ADRs,
and `docs/limitations.md` for known boundaries.
