#!/usr/bin/env bash
# Initialize the database using Alembic migrations.
# Run this once before starting the application in production.
set -euo pipefail

echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete."
