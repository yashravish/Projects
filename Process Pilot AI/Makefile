.PHONY: up down seed test dev-backend dev-frontend install migrate logs clean

up:
	docker-compose up --build -d

down:
	docker-compose down

logs:
	docker-compose logs -f

seed:
	cd backend && python -m app.seed

test:
	cd backend && python -m pytest -v

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

migrate:
	cd backend && alembic upgrade head

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
