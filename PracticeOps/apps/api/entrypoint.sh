#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head

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
"

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
