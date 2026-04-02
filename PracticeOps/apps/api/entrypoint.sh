#!/bin/bash

MAX_RETRIES=5
RETRY_DELAY=3

echo "Running database migrations..."
for i in $(seq 1 $MAX_RETRIES); do
    if alembic upgrade head; then
        echo "Migrations completed successfully."
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "WARNING: Migrations failed after $MAX_RETRIES attempts. Starting server anyway."
    else
        echo "Migration attempt $i/$MAX_RETRIES failed. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    fi
done

echo "Checking if database needs demo data..."
python -c "
import asyncio
from app.database import async_session_maker
from sqlalchemy import select, func
from app.models import User

async def check_and_seed():
    async with async_session_maker() as db:
        result = await db.execute(select(func.count()).select_from(User))
        count = result.scalar()
        if count == 0:
            print('Database is empty, seeding demo data...')
            import subprocess
            subprocess.run(['python', '-m', 'scripts.seed_demo'], check=True)
        else:
            print(f'Database already has {count} users, skipping seed')

asyncio.run(check_and_seed())
" || echo "WARNING: Demo data check failed. Starting server anyway."

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
