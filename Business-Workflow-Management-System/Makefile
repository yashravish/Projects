# Business Workflow Management System - common developer commands

.PHONY: help up down logs build seed migrate test test-backend test-frontend test-e2e \
        backend-shell frontend-shell db-shell clean

help:
	@echo "Available targets:"
	@echo "  make up              - Start the full stack via Docker Compose"
	@echo "  make down            - Stop and remove containers"
	@echo "  make logs            - Tail logs from all services"
	@echo "  make build           - Rebuild Docker images"
	@echo "  make migrate         - Apply Alembic migrations inside the backend container"
	@echo "  make seed            - Seed demo data inside the backend container"
	@echo "  make test            - Run backend + frontend test suites"
	@echo "  make test-backend    - Run pytest"
	@echo "  make test-frontend   - Run Vitest"
	@echo "  make test-e2e        - Run Playwright tests"
	@echo "  make backend-shell   - Shell into the backend container"
	@echo "  make frontend-shell  - Shell into the frontend container"
	@echo "  make db-shell        - Open psql against the database"
	@echo "  make clean           - Remove volumes and orphans"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

migrate:
	docker compose exec backend alembic upgrade head

seed:
	docker compose exec backend python -m app.db.seed

test: test-backend test-frontend

test-backend:
	docker compose exec backend pytest

test-frontend:
	docker compose exec frontend npm run test

test-e2e:
	cd frontend && npm run test:e2e

backend-shell:
	docker compose exec backend bash

frontend-shell:
	docker compose exec frontend sh

db-shell:
	docker compose exec db psql -U workflow -d workflow

clean:
	docker compose down -v --remove-orphans
