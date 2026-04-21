.PHONY: up down test migrate logs clean

up:
	docker compose up --build -d

down:
	docker compose down

test:
	docker compose exec api pytest -v

migrate:
	docker compose exec api alembic upgrade head

logs:
	docker compose logs -f api

clean:
	docker compose down -v
