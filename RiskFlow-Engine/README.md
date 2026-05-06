# RiskFlow Engine

A real-time risk calculation platform for simulated equity and fixed-income portfolios, using Java/Spring Boot, Python risk analytics, PostgreSQL, Redis, and Redis Streams.

## Project structure

```text
RiskFlow-Engine/
├── .env.example
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pom.xml
├── README.md
├── src/main/java/com/riskflow/
│   ├── RiskFlowApplication.java
│   ├── config/RedisConfig.java
│   ├── controller/ApiController.java
│   ├── dto/Requests.java
│   ├── dto/Responses.java
│   ├── exception/GlobalExceptionHandler.java
│   ├── exception/NotFoundException.java
│   ├── model/*.java
│   ├── repository/*.java
│   └── service/*.java
├── src/main/resources/application.yml
├── src/test/java/com/riskflow/
│   ├── controller/ApiControllerTest.java
│   └── service/*Test.java
└── risk-worker/
    ├── Dockerfile
    ├── requirements.txt
    └── risk_engine/
        ├── bond_math.py
        ├── calculations.py
        ├── equity_math.py
        ├── stress.py
        ├── worker.py
        └── tests/test_calculations.py
```

## Why this matters for Risk Technology

RiskFlow Engine demonstrates common hedge-fund risk-technology patterns: low-latency REST APIs for trade capture and risk queries, durable SQL storage for auditability, Redis caching for latest risk views, event-driven market-data fanout through Redis Streams, and separate Python analytics code mirroring front-office quant calculations.

## Architecture

```text
Market Data Simulator
        |
        v
Redis Streams: market-data
        |
        v
Spring Boot Risk API  <----> PostgreSQL
        |
        v
Redis Cache
        |
        v
Python Risk Worker / Risk Calculation Module
```

PostgreSQL stores portfolios, instruments, trades, market prices, risk results, EOD reports, and audit logs. Redis stores keys such as `price:AAPL`, `portfolio:1:risk:latest`, `portfolio:1:exposure`, and `portfolio:1:alerts`.

## Tech stack

- Java 21, Spring Boot 3, Spring Web, Spring Data JPA, Bean Validation, Maven
- PostgreSQL 16 for durable records
- Redis 7 for cache and Redis Streams messaging
- Python 3.12, Pandas, NumPy, pytest, requests
- Docker Compose for local orchestration
- GitHub Actions for Java and Python tests

## Setup with Docker

```bash
cp .env.example .env
docker compose up --build
```

Verify the API:

```bash
curl http://localhost:8080/api/health
```

Expected response:

```json
{"service":"RiskFlow Engine","status":"UP"}
```

## Setup without Docker

Start PostgreSQL and Redis locally, then create a database/user matching `.env.example`, or override these variables:

```bash
export SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/riskflow
export SPRING_DATASOURCE_USERNAME=riskflow
export SPRING_DATASOURCE_PASSWORD=riskflow
export SPRING_REDIS_HOST=localhost
export SPRING_REDIS_PORT=6379
mvn spring-boot:run
```

Run the Python analytics tests locally:

```bash
cd risk-worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_DB` | `riskflow` | PostgreSQL database used by Docker Compose |
| `POSTGRES_USER` | `riskflow` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `riskflow` | PostgreSQL password |
| `SPRING_DATASOURCE_URL` | `jdbc:postgresql://postgres:5432/riskflow` | JDBC URL in Docker |
| `SPRING_DATASOURCE_USERNAME` | `riskflow` | Spring DB username |
| `SPRING_DATASOURCE_PASSWORD` | `riskflow` | Spring DB password |
| `SPRING_REDIS_HOST` | `redis` | Redis host in Docker |
| `SPRING_REDIS_PORT` | `6379` | Redis port |
| `API_BASE_URL` | `http://spring-api:8080` | Python worker API target |

## Database schema overview

- `portfolios`: book identifier and creation time.
- `instruments`: equity/bond reference data with coupon, maturity, and face value for bonds.
- `trades`: portfolio/instrument trade ledger with side, quantity, price, and trade time.
- `market_prices`: durable market price/yield history.
- `risk_results`: historical risk snapshots with exposure, PnL, VaR, stress, DV01, and concentration.
- `eod_reports`: EOD summaries derived from the latest risk result.
- `audit_logs`: operational audit trail for trades, market data, risk, and EOD events.

Hibernate creates schema automatically for local/demo use. In production, convert the entity model to controlled Flyway/Liquibase migrations.

## API endpoints

| Area | Method | Path |
| --- | --- | --- |
| Health | GET | `/api/health` |
| Portfolios | POST/GET | `/api/portfolios` |
| Portfolios | GET | `/api/portfolios/{id}` |
| Instruments | POST/GET | `/api/instruments` |
| Instruments | GET | `/api/instruments/{id}` |
| Trades | POST/GET | `/api/trades` |
| Trades | GET | `/api/portfolios/{portfolioId}/trades` |
| Market Data | POST | `/api/market-prices` |
| Market Data | GET | `/api/instruments/{instrumentId}/prices/latest` |
| Risk | POST | `/api/portfolios/{portfolioId}/risk/calculate` |
| Risk | GET | `/api/portfolios/{portfolioId}/risk/latest` |
| Risk | GET | `/api/portfolios/{portfolioId}/risk/history` |
| EOD | POST | `/api/portfolios/{portfolioId}/eod` |
| EOD | GET | `/api/portfolios/{portfolioId}/eod/latest` |
| Demo | POST | `/api/demo/seed` |
| Demo | POST | `/api/demo/run-market-simulation` |

## Demo walkthrough

```bash
docker compose up --build
curl -X POST http://localhost:8080/api/demo/seed
curl -X POST http://localhost:8080/api/demo/run-market-simulation
curl -X POST http://localhost:8080/api/portfolios/1/risk/calculate
curl http://localhost:8080/api/portfolios/1/risk/latest
curl -X POST http://localhost:8080/api/portfolios/1/eod
curl http://localhost:8080/api/portfolios/1/eod/latest
```

## Sample curl commands

Create a portfolio:

```bash
curl -X POST http://localhost:8080/api/portfolios \
  -H 'Content-Type: application/json' \
  -d '{"name":"Convertible Arbitrage"}'
```

Create an equity instrument:

```bash
curl -X POST http://localhost:8080/api/instruments \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"AAPL","type":"EQUITY","name":"Apple Inc."}'
```

Create a bond instrument:

```bash
curl -X POST http://localhost:8080/api/instruments \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"UST10Y","type":"BOND","name":"US Treasury 10Y","couponRate":0.035,"maturityDate":"2036-05-06","faceValue":1000}'
```

Submit a trade:

```bash
curl -X POST http://localhost:8080/api/trades \
  -H 'Content-Type: application/json' \
  -d '{"portfolioId":1,"instrumentId":1,"side":"BUY","quantity":10000,"price":185.00}'
```

Post market data:

```bash
curl -X POST http://localhost:8080/api/market-prices \
  -H 'Content-Type: application/json' \
  -d '{"instrumentId":1,"price":182.50}'
```

## Example risk report output

```json
{
  "portfolioId": 1,
  "totalMarketValue": 4820000.00,
  "totalPnL": -22400.00,
  "equityExposure": 3620000.00,
  "fixedIncomeExposure": 1200000.00,
  "delta": 3620000.00,
  "dv01": -4300.00,
  "var95": 96400.00,
  "stressEquityDown5": -181000.00,
  "stressEquityDown10": -362000.00,
  "stressRatesUp25bps": -107500.00,
  "stressRatesUp100bps": -430000.00,
  "concentrationPct": 41.25
}
```

## Risk metric explanations

- **Position quantity**: buy quantity minus sell quantity.
- **Market value**: position quantity multiplied by latest market price for equities; quantity multiplied by simplified present-value bond price for fixed income.
- **PnL**: position quantity times latest price minus average trade price.
- **Delta**: equity exposure proxy used for first-order equity sensitivity.
- **DV01**: approximate bond price change for a one-basis-point yield increase. It is usually negative for long fixed-rate bonds.
- **VaR 95**: fallback estimate equal to 2% of absolute portfolio market value when there is insufficient history.
- **Stress equity down 5/10%**: loss estimate from equity shocks.
- **Stress rates up 25/100 bps**: loss estimate from DV01 multiplied by the rate shock.
- **Concentration percent**: largest absolute instrument exposure divided by absolute portfolio market value.

## Low-latency and production-readiness mapping

- The risk endpoint logs calculation latency and is structured so p50/p95 metrics can be exported to Micrometer/Prometheus.
- Repository indexes target the most common risk queries: portfolio trade history, market price lookup by instrument/time, risk history by portfolio/time, and EOD lookup by portfolio/date.
- Redis separates latest-state reads from PostgreSQL durability, reflecting a SQL/NoSQL design common in intraday risk systems.
- Redis Streams provides a simple event-driven market-data path without needing a heavyweight Kafka cluster for local demos.
- Audit logs create operational traceability for trade capture, price updates, risk calculation, and EOD generation.
- Java implements service orchestration and API reliability, while Python mirrors risk formulas in an analytics-friendly stack.

## Testing instructions

Java tests:

```bash
mvn test
```

Python tests:

```bash
cd risk-worker
pip install -r requirements.txt
pytest -q
```

Docker build check:

```bash
docker compose build
```

## CI/CD

The GitHub Actions workflow checks out the repository, installs Java 21 and Python 3.12, runs Maven tests, installs Python dependencies, runs pytest, and builds Docker images.

## Troubleshooting

- **API cannot connect to PostgreSQL**: confirm `docker compose ps` shows `postgres` healthy and that `SPRING_DATASOURCE_URL` uses host `postgres` inside Docker.
- **API cannot connect to Redis**: confirm Redis is healthy and `SPRING_REDIS_HOST=redis` for Docker or `localhost` for non-Docker.
- **`portfolio 1` not found**: run `curl -X POST http://localhost:8080/api/demo/seed` first. If you reused an old database volume, run `docker compose down -v` to reset IDs.
- **Maven cannot resolve dependencies**: verify internet access to Maven Central or use a corporate Maven mirror in `~/.m2/settings.xml`.
- **Python import errors**: run `pip install -r risk-worker/requirements.txt` from the `risk-worker` directory.

## Future improvements

- Replace `ddl-auto=update` with Flyway migrations.
- Add Micrometer Prometheus metrics for p50/p95/p99 latency.
- Add Redis Stream consumers that automatically recalculate affected portfolios.
- Add Testcontainers-backed PostgreSQL and Redis integration tests in CI.
- Implement historical simulation VaR from market-price return histories.
- Add authentication and role-based access control.
- Add OpenAPI documentation.

## Resume bullets tailored to ExodusPoint Risk Technology

- Built RiskFlow Engine, a Java 21/Spring Boot real-time risk platform with PostgreSQL durability, Redis cache/Streams messaging, Docker Compose orchestration, and Python analytics tests.
- Implemented equity and fixed-income risk calculations including market value, PnL, delta, DV01, VaR fallback, stress scenarios, EOD reporting, concentration risk, and audit logging.
- Designed low-latency latest-risk reads with Redis keys and durable portfolio/trade/risk history in indexed PostgreSQL tables.
- Created portfolio-ready CI/CD with Maven, pytest, Docker builds, layered services/controllers/repositories, validation, exception handling, and automated unit/integration tests.
