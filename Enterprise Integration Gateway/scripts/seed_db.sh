#!/usr/bin/env bash
# Seed the database by triggering a full sync via the API.
# Requires the application to be running.
set -euo pipefail

APP_URL="${APP_URL:-http://localhost:8000}"

echo "Seeding database via full sync..."
curl -s -X POST "$APP_URL/api/v1/sync/all" | python3 -m json.tool
echo "Done. Check $APP_URL/api/v1/admin/status for record counts."
