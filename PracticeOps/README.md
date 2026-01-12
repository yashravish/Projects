# PracticeOps

Practice management for musical teams — a cappella groups, choirs, and ensembles.

## Quick Start

```bash
# Start all services (Postgres, API, Web)
make dev
```

That's it. Visit:
- **Web App**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Architecture

```
PracticeOps/
├── apps/
│   ├── api/          # FastAPI backend
│   └── web/          # React frontend
├── infra/
│   └── docker-compose.yml
└── docs/
    ├── CHECKLIST.md  # Milestone progress
    └── CONTRIBUTING.md
```

## Development

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.11+ for local API development
- (Optional) Node.js 20+ for local web development

### Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start all services |
| `make dev-build` | Rebuild and start |
| `make down` | Stop all services |
| `make logs` | View logs |
| `make test` | Run all tests |
| `make lint` | Run all linters |

### Local Development (without Docker)

**API:**
```bash
cd apps/api
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

**Web:**
```bash
cd apps/web
npm install
npm run dev
```

## Testing

```bash
# All tests
make test

# API only
make test-api

# Web only
make test-web

# E2E tests
cd apps/web && npx playwright test
```

## Tech Stack

**Backend:**
- FastAPI
- SQLAlchemy 2.0 (async)
- Alembic
- Pydantic v2
- PostgreSQL

**Frontend:**
- React 18
- TypeScript
- Vite
- TanStack Query
- React Router v6
- Tailwind CSS
- shadcn/ui

## Deployment

Ready to ship? See [docs/deployment.md](docs/deployment.md) for full instructions.

**Quick deploy to Railway:**
```bash
npm install -g @railway/cli
railway login
railway init
railway add --database postgres
# Deploy API
cd apps/api && railway up
# Deploy Web
cd ../web && railway up
```

**Environment variables:** Copy `env.example` to `.env` and configure for your environment.

## License

MIT

