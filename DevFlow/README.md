# DevFlow: AI-Powered CI/CD + Release Intelligence Platform

DevFlow is a **full-stack portfolio-grade developer platform** that simulates CI/CD pipelines, tracks deployments, ships feature flags and A/B experiments, exposes **Prometheus-style** metrics, and offers an **OpenAI-powered failure analysis** with a **safe mock fallback** when no API key is set.

## Architecture

```mermaid
flowchart LR
  subgraph Client
    U[React + Vite + Tailwind]
  end
  subgraph API[FastAPI]
    R[Routers]
    S[Services: pipeline, deploy, AI]
    M[In-memory + DB metrics]
  end
  DB[(PostgreSQL)]
  U -->|HTTP JSON| R
  R --> S
  S --> DB
  R --> M
  OpenAI[(OpenAI API)] -. optional .-> S
```

## Features

- **Projects** with slug, description, and relationships to runs and deployments
- **Pipeline simulation**: lint → unit → integration → build → deploy, with per-stage logs, duration, and deterministic sample failures
- **Tests**: recording results against runs
- **Deployments**: canary (10/25/50/100) with auto-rollback on simulated high error rate
- **Feature flags**: CRUD, rollout %, and deterministic user evaluation via hashing
- **A/B testing**: create experiments, assign users deterministically, ingest metrics, view aggregates
- **Defects**: full defect tracking, stats (open/resolved, rate, by severity), optional auto-creation from AI analysis, KB article links
- **Knowledge base**: articles (runbook, postmortem, pipeline, troubleshooting) with bi-directional links to defects and AI reports
- **AI failure analyzer**: structured root cause, component, fix, severity, confidence; OpenAI if key is present; rule-based **mock** otherwise
- **Metrics**: `GET /metrics` (Prometheus text), `GET /api/dashboard/metrics` (JSON for UI)

## Tech stack

| Area        | Technology |
|------------|------------|
| API        | Python 3.12, FastAPI, Pydantic v2, Uvicorn |
| Data       | SQLAlchemy 2.x (async), Alembic, PostgreSQL |
| UI         | React 18, TypeScript, Vite, Tailwind CSS |
| Ops        | Docker, Docker Compose, GitHub Actions |
| AI         | OpenAI (optional) + built-in heuristics |
| Tests      | Pytest, Vitest, React Testing Library |

## Project layout

- `backend/` — FastAPI app, Alembic, pytest
- `frontend/` — React app, Vitest
- `docker-compose.yml` — postgres + backend + frontend (reverse proxy to API)
- `gcp/` — Cloud Build + helper scripts, Artifact Registry naming
- `terraform/` — sample GCP: Artifact Registry, Cloud SQL, Secret Manager
- `observability/` — example Prometheus config (optional)

## Quick start (local)

1. **Clone / open the `devflow` directory** (treat it as the git root for CI paths).

2. **Copy environment**
   - `cp .env.example .env` and adjust if needed.
   - Set `OPENAI_API_KEY` only if you want live LLM analysis (optional).

3. **Docker (recommended)**
   ```bash
   docker compose up --build
   ```
   - API: <http://localhost:8000/docs>
   - UI (Nginx, proxies `/api` to API): <http://localhost:5173> in compose is mapped to **port 80** in the `frontend` service — in `docker-compose.yml` we expose `5173:80`, so open **<http://localhost:5173>**

4. **Seed data (optional)**
   ```bash
   cd backend
   pip install -r requirements.txt
   set DATABASE_URL=postgresql+asyncpg://devflow:devflow@localhost:5432/devflow   # Windows PowerShell: $env:DATABASE_URL=...
   alembic upgrade head
   python -m scripts.seed_data
   ```

5. **Dev without full Docker UI**
   - Start Postgres: `docker compose up -d postgres`
   - Backend: `cd backend && alembic upgrade head && uvicorn app.main:app --reload --port 8000`
   - Frontend: `cd frontend && npm install && npm run dev` (Vite proxies `/api` to port 8000)

## API overview (selected)

| Method & path | Purpose |
|---------------|--------|
| `GET/POST /api/projects` | List / create projects |
| `POST /api/pipelines/{id}/trigger` | Run full simulation |
| `GET /api/pipelines/by-project/{id}` | History |
| `POST /api/deployments/{id}` + `/canary`, `/rollback` | Deploy, canary, rollback |
| `CRUD /api/flags` + `POST /api/flags/evaluate` | Feature flags + evaluation |
| `.../api/experiments` + assign/metrics/aggregate | A/B |
| `POST /api/ai/analyze` | AI / mock log analysis, optional defect |
| `GET/POST/PATCH` `/api/defects` + `GET /api/defects/stats` | Defects |
| `.../api/kb` | Knowledge base |
| `GET /metrics` | Prometheus text |
| `GET /api/dashboard/metrics` | JSON metrics |

## Screenshots (placeholders for portfolio)

| Area | Suggestion |
|------|------------|
| Dashboard | Capture overview cards and project list |
| Pipelines | Table with stage trace + trigger button |
| AI Analyzer | Log textarea + result panel |
| Defects | Stats row + open defects table |

## Google Cloud Run (summary)

- Build and push: see `gcp/build_push.sh` and `gcp/cloudbuild.yaml`.
- Image URI pattern: `REGION-docker.pkg.dev/PROJECT/REPO/NAME:TAG` (e.g. `us-central1-docker.pkg.dev/PROJECT_ID/devflow/devflow-api:2026-04-26`).
- **Cloud SQL**: create a Postgres instance, set `DATABASE_URL` for async SQLAlchemy (`postgresql+asyncpg://...` with the Cloud SQL Python connector or public IP, depending on your network).
- **Secrets**: store `OPENAI_API_KEY` in Secret Manager; reference from Cloud Run.
- `terraform/main.tf` provisions Artifact Registry, Cloud SQL, and a sample Secret — extend with real IAM, VPC, and Cloud Run wiring for production.

## Resume bullets (polished)

- Built an **AI-powered CI/CD and release intelligence platform** using **FastAPI**, **React**, **PostgreSQL**, **Docker**, and **GitHub Actions**.
- Implemented **pipeline simulation**, **feature flags**, **A/B testing**, **canary deployments**, **rollback logic**, and **Prometheus-style observability** metrics.
- Integrated **OpenAI-powered log analysis** with **fallback rule-based diagnostics** to surface root cause, severity, confidence, and suggested fixes.
- Developed **automated backend and frontend test suites** and **CI** workflows to validate API behavior, simulation logic, and key UI surface areas.

## Future improvements

- Real SCM webhooks, artifact storage, and agent-based remediation
- Fine-grained RBAC, audit log, and multi-tenancy
- OpenTelemetry export alongside Prometheus
- Durable A/B user assignments and idempotent metrics ingestion
- Hardening Terraform: IAM, VPC, managed SSL, and Cloud SQL private IP

## License

Use freely for **portfolio** and **learning**. Review cloud costs before `terraform apply`.
