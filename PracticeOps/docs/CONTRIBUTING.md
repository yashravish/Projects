# Contributing to PracticeOps

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 20+
- Python 3.11+

### Quick Start
```bash
# Clone the repository
git clone <repo-url>
cd PracticeOps

# Start all services
make dev
```

This starts:
- **Postgres** on port 5432
- **API** on port 8000
- **Web** on port 5173

## Running Tests

### API Tests
```bash
# Run all API tests
make test-api

# Or manually
cd apps/api
pip install -e ".[dev]"
pytest -v
```

### Web Tests
```bash
# Run unit tests
make test-web

# Or manually
cd apps/web
npm install
npm test
```

### E2E Tests
```bash
# Run Playwright tests
cd apps/web
npx playwright install
npx playwright test
```

## Linting & Formatting

### API
```bash
# Check linting
cd apps/api
ruff check .
mypy .

# Auto-format
ruff format .
ruff check --fix .
```

### Web
```bash
# Check linting
cd apps/web
npm run lint
npm run typecheck

# Auto-format
npm run format
```

## CI Pipeline

The CI workflow runs on every push and PR to `main`:

1. **Lint** — Ruff, ESLint
2. **Typecheck** — Mypy, TypeScript
3. **Test** — pytest, Vitest

### CI Expectations
- All checks must pass before merge
- No warnings allowed in lint output
- 100% of tests must pass
- Type errors are blocking

## Code Standards

### Python (API)
- Follow PEP 8 (enforced by Ruff)
- Use type hints everywhere
- Async by default for I/O operations
- Pydantic models for all request/response schemas

### TypeScript (Web)
- Strict mode enabled
- No `any` types without justification
- React hooks rules enforced
- Prettier formatting required

## Branch Strategy

- `main` — production-ready code
- Feature branches — `feature/<milestone>-<description>`
- Bugfix branches — `fix/<description>`

## Commit Messages

Use conventional commits:
```
feat(api): add user authentication endpoints
fix(web): correct form validation on login
docs: update CONTRIBUTING.md
test(api): add edge case tests for ticket status
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes with tests
3. Ensure CI passes locally: `make lint test`
4. Open PR with clear description
5. Address review comments
6. Squash and merge when approved

## Architecture Decisions

### API Layer
- FastAPI with async SQLAlchemy 2.0
- Alembic for migrations
- Pydantic v2 for validation
- RBAC enforced at every endpoint

### Frontend
- React 18 with TypeScript
- TanStack Query for server state
- React Router v6 for routing
- shadcn/ui for components
- Tailwind CSS for styling

### Testing Strategy
- Unit tests for business logic
- Integration tests for API endpoints
- E2E tests for critical user flows
- No mocking of database in API tests (use test DB)

## Getting Help

- Check existing issues first
- Create a new issue with reproduction steps
- Use discussions for questions

