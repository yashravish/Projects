# PublicSector Agentic Document Intelligence Platform — developer commands.
# All targets are run against docker compose unless noted.

SHELL := /bin/sh
COMPOSE ?= docker compose

.PHONY: help up down build logs ps psql redis-cli api-shell test lint typecheck migrate revision seed eval clean

help:
	@echo "Targets:"
	@echo "  up           - docker compose up --build (foreground)"
	@echo "  up-d         - docker compose up --build -d"
	@echo "  down         - docker compose down"
	@echo "  build        - docker compose build"
	@echo "  logs         - tail logs for api"
	@echo "  ps           - list services"
	@echo "  psql         - psql into postgres"
	@echo "  redis-cli    - redis-cli into redis"
	@echo "  api-shell    - bash into api container"
	@echo "  test         - pytest (in api container) with coverage gate"
	@echo "  lint         - ruff + mypy (in api container)"
	@echo "  typecheck    - mypy --strict app (in api container)"
	@echo "  migrate      - alembic upgrade head (in api container)"
	@echo "  revision m=  - alembic revision --autogenerate -m \"\$\$m\""
	@echo "  seed         - python -m app.seed.seed_data (in api container)"
	@echo "  eval         - run evaluation harness against the seeded dataset"
	@echo "  clean        - down -v (DESTROYS volumes)"

up:
	$(COMPOSE) up --build

up-d:
	$(COMPOSE) up --build -d

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f api

ps:
	$(COMPOSE) ps

psql:
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-psdi} -d $${POSTGRES_DB:-psdi}

redis-cli:
	$(COMPOSE) exec redis redis-cli

api-shell:
	$(COMPOSE) exec api bash

test:
	$(COMPOSE) exec api pytest --cov=app --cov-report=term-missing --cov-fail-under=75

lint:
	$(COMPOSE) exec api ruff check app tests
	$(COMPOSE) exec api mypy app

typecheck:
	$(COMPOSE) exec api mypy --strict app

migrate:
	$(COMPOSE) exec api alembic upgrade head

revision:
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(m)"

seed:
	$(COMPOSE) exec api python -m app.seed.seed_data

eval:
	$(COMPOSE) exec api python -m app.evaluation.runner --dataset default --model gpt-4o-mini --top-k 8 --rerank true --tag baseline

clean:
	$(COMPOSE) down -v
