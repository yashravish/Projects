#!/usr/bin/env bash
# Container entrypoint: wait for postgres, run migrations, optionally seed,
# then exec the supplied command (uvicorn or celery).
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

PG_HOST="${POSTGRES_HOST:-postgres}"
PG_PORT="${POSTGRES_PORT:-5432}"

log "waiting for postgres at ${PG_HOST}:${PG_PORT}..."
ATTEMPTS=0
until nc -z "${PG_HOST}" "${PG_PORT}" >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "${ATTEMPTS}" -ge 60 ]; then
        log "postgres did not become reachable after 60 attempts"
        exit 1
    fi
    sleep 1
done
log "postgres is reachable"

# Run migrations (and optional seed) for API, test runs, and anything invoked
# as ``python -m ...``. Skip for one-off admin commands / celery so workers
# never race the migrator; ``pytest`` alone also migrates.
case "${1:-}" in
    uvicorn|python|fastapi|pytest)
        log "running alembic upgrade head"
        alembic upgrade head

        if [ "${SEED_ON_BOOT:-false}" = "true" ]; then
            log "running seed_data"
            python -m app.seed.seed_data || log "seed_data exited non-zero (continuing)"
        fi
        ;;
    *)
        log "non-API command (${1:-}); skipping migrations/seed"
        ;;
esac

log "exec: $*"
exec "$@"
