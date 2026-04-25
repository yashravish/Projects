# PublicSector Agentic Document Intelligence Platform

A responsible-AI document intelligence system for federal/public-sector
analysts. Upload public-sector PDFs (grant guidance, procurement rules,
policy documents, compliance manuals, benefit program docs, agency reports);
the platform ingests, chunks, embeds, indexes, retrieves evidence via hybrid
search, reasons via a multi-agent LangGraph workflow, validates citations
against source text, flags risks, tracks evaluation metrics in MLflow, and
generates system-card reports.

This is **not a chatbot over PDFs** — it is a grounded, audited, evaluated
document intelligence system where every answer is citation-backed,
uncertainty-aware, and observable.

> **Status:** under active staged construction. Stage 1 (auth + skeleton +
> health) is complete and bootable. Subsequent stages (ingestion, retrieval,
> agent graph, evaluation, system card, cloud ML) are tracked in
> [`docs/decisions.md`](docs/decisions.md) and built incrementally.

---

## Quickstart

```bash
cp .env.example .env
# Optionally set OPENAI_API_KEY in .env (not required for Stage 1).
docker compose up --build
```

Once boot completes:

| Service      | URL                              |
| ------------ | -------------------------------- |
| API          | http://localhost:8000            |
| API docs     | http://localhost:8000/docs       |
| Health       | http://localhost:8000/health     |
| MLflow       | http://localhost:5000            |
| Postgres     | `localhost:5432` (psdi/psdi_dev_password) |
| Redis        | `localhost:6379`                 |

### Seeded credentials (development only)

Seeded by `app/seed/seed_data.py` when `SEED_ON_BOOT=true`:

- Email: `seed-admin@example.gov`
- Password: `ChangeMe!2026`
- Organization: `Demo Agency`

```bash
# Sanity check
curl -s http://localhost:8000/health | jq
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"seed-admin@example.gov","password":"ChangeMe!2026"}' | jq
```

---

## Development

```bash
make up          # docker compose up --build (foreground)
make logs        # tail api logs
make psql        # psql into postgres
make migrate     # alembic upgrade head inside the api container
make test        # pytest with coverage gate
make lint        # ruff + mypy
```

Local dev without Docker:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Point at a local Postgres + Redis, then:
alembic upgrade head
uvicorn app.main:app --reload
```

---

## Architecture (planned)

```
client (React/Vite)
  └── HTTPS ──► FastAPI (uvicorn)
                  ├── auth (JWT)                ─► Postgres (users, orgs)
                  ├── documents/upload          ─► S3/local + Celery
                  ├── celery worker             ─► PyMuPDF → chunker → embeddings → Postgres (chunks: vector + tsvector)
                  ├── /query                    ─► hybrid retrieval (dense + FTS+BM25 + RRF)
                  │                                ─► reranker (local cross-encoder | SageMaker endpoint)
                  │                                ─► LangGraph: retriever → reasoning → validator → risk → report
                  ├── /evaluations              ─► gold dataset → metrics → MLflow
                  ├── /analytics                ─► aggregates from query_runs
                  └── /system-card              ─► Jinja2 markdown
```

Detailed diagrams: `docs/architecture-diagram.svg` (added in a later stage).

---

## Tenant isolation

Every tenant-scoped table carries `organization_id` with an index. The
`apply_tenant_filter` helper in `app/security/tenant.py` is the canonical way
to scope a query; a unit test (`tests/unit/test_tenant_isolation.py`, added in
Stage 3) uses AST inspection to fail the build if any service method touching
tenant tables omits `organization_id` from its signature.

Tokens carry `org` claim alongside `sub` so the boundary is enforced both at
the route layer (via `Depends(get_current_user)`) and the query layer.

---

## Responsible AI statement

This platform is designed to support — not replace — human analysts. Every
answer must be reviewed by a domain expert before acting on it. The system:

- Refuses to provide legal, financial, or compliance advice.
- Cites every claim and quantifies grounding strength on every response.
- Flags ambiguity, conflicting sources, scope mismatch, and insufficient
  evidence as first-class outputs.
- Logs every model and prompt version on every run for auditability.

See the system-card endpoint (added in Stage 4) for the per-deployment card.

---

## License

Apache-2.0.
