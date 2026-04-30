#!/bin/sh
set -e

echo "[entrypoint] Waiting for database..."
python - <<'PY'
import os, time
from sqlalchemy import create_engine
url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://workflow:workflow@db:5432/workflow")
for attempt in range(60):
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        print("[entrypoint] Database is ready")
        break
    except Exception as e:
        print(f"[entrypoint] DB not ready ({attempt+1}/60): {e}")
        time.sleep(1)
else:
    raise SystemExit("[entrypoint] Database unavailable after 60s")
PY

echo "[entrypoint] Running Alembic migrations..."
alembic upgrade head

echo "[entrypoint] Seeding database..."
python -m app.db.seed || true

echo "[entrypoint] Starting Uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
