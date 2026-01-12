.PHONY: dev dev-build down logs api-shell web-shell db-shell test test-api test-web lint lint-api lint-web format typecheck clean

# =============================================================================
# LOCAL DEVELOPMENT
# =============================================================================

dev:
	docker compose -f infra/docker-compose.yml up

dev-build:
	docker compose -f infra/docker-compose.yml up --build

down:
	docker compose -f infra/docker-compose.yml down

logs:
	docker compose -f infra/docker-compose.yml logs -f

# =============================================================================
# SHELL ACCESS
# =============================================================================

api-shell:
	docker compose -f infra/docker-compose.yml exec api bash

web-shell:
	docker compose -f infra/docker-compose.yml exec web sh

db-shell:
	docker compose -f infra/docker-compose.yml exec postgres psql -U practiceops -d practiceops

# =============================================================================
# TESTING
# =============================================================================

test: test-api test-web

test-api:
	cd apps/api && pytest -v

test-web:
	cd apps/web && npm test

test-e2e:
	cd apps/web && npx playwright test

# =============================================================================
# LINTING & FORMATTING
# =============================================================================

lint: lint-api lint-web

lint-api:
	cd apps/api && ruff check . && mypy .

lint-web:
	cd apps/web && npm run lint

format:
	cd apps/api && ruff format . && ruff check --fix .
	cd apps/web && npm run format

typecheck:
	cd apps/api && mypy .
	cd apps/web && npm run typecheck

# =============================================================================
# DATABASE
# =============================================================================

migrate:
	cd apps/api && alembic upgrade head

migrate-create:
	@read -p "Migration name: " name; cd apps/api && alembic revision --autogenerate -m "$$name"

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	docker compose -f infra/docker-compose.yml down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true

