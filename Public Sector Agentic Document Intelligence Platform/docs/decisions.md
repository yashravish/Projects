# Architecture Decision Records

Format: lightweight ADRs (Context → Decision → Consequences). Newest at top.

---

## ADR-005: Stage 1 scope — auth + skeleton + health

**Date:** 2026-04-25
**Status:** Accepted

### Context

The platform is being built in 5 stages so each stage is verifiable end-to-end
before the next is added. Stage 1 is the bootable skeleton.

### Decision

Stage 1 ships:

- Compose stack: Postgres (pgvector), Redis, MLflow, API.
- Backend: config + structlog + async SQLAlchemy + Alembic with the **full
  schema for the entire platform** (organizations, users, documents, chunks
  with `vector(1536)` + generated `tsvector`, query_runs, evaluation_runs,
  system_cards, audit_logs).
- Auth: register / login / refresh / me with HS256/RS256 JWT.
- Health: probes db, redis, mlflow, openai presence.
- Idempotent seed of demo org + admin user.
- Tests: unit (passwords, JWT, config) + integration (auth flow, health) over
  a real Postgres.
- CI: ruff + mypy + pytest with coverage gate (40% in Stage 1, raised to 75%
  in Stage 5) + `docker compose build`.

Routes / services / frontend for ingestion, retrieval, agent, evaluation,
analytics, and system card are intentionally not in Stage 1.

### Consequences

- The system boots and is loginable on first `docker compose up --build`.
- Schema for all later stages is already in place, so no destructive
  migrations later.
- Coverage gate is temporarily lowered; the contract is that Stage 5 raises it
  to the spec'd 75%.

---

## ADR-004: Use synthetic public-domain-style PDFs for the gold dataset

**Date:** 2026-04-25
**Status:** Accepted (will be implemented in Stage 4)

### Context

The platform is positioned around real public-sector documents (NIH NOFOs, FAR
Part 15, OMB A-11, SBA 7(a)). Bundling those texts directly raises licensing
and provenance ambiguity.

### Decision

Generate 3–4 synthetic PDFs at image-build time using `reportlab`, with
realistic public-sector style and section structure written specifically for
this repo. The 25-item gold dataset references these synthetic documents.

### Consequences

- Deterministic and license-clean.
- Evaluation numbers are not directly comparable to other systems on real
  public datasets, but they're internally consistent for regression testing.

---

## ADR-003: PostgreSQL FTS for keyword recall, BM25 for in-set ranking

**Date:** 2026-04-25
**Status:** Accepted (will be implemented in Stage 3)

### Context

The spec calls for BM25 keyword retrieval. Native Postgres FTS uses
`ts_rank_cd`, which is similar in spirit to BM25 but not identical. Running
BM25 over the entire corpus would require either a dedicated index (Lucene,
Tantivy) or in-memory recomputation per query, which doesn't scale.

### Decision

Use Postgres FTS (`plainto_tsquery` against the `tsv` generated column,
ordered by `ts_rank_cd`) to retrieve the top-30 keyword candidates per query.
Then run `rank-bm25` **over only that candidate set** to refine the ordering
before fusion.

### Consequences

- We get FTS recall without a separate search engine.
- BM25 controls the ranking that matters most (the head of the result set).
- The candidate-set BM25 is in-memory and bounded (≤30 docs × O(query_terms)).

---

## ADR-002: Local + SageMaker reranker behind a single interface

**Date:** 2026-04-25
**Status:** Accepted (will be implemented in Stage 5)

### Context

Reviewers without AWS credentials must still be able to run the platform
end-to-end. SageMaker is required by the project's cloud-ML brief.

### Decision

`app/services/reranker_service.py` exposes a single `Reranker` Protocol with
two implementations selected by `RERANKER_BACKEND` (`local` |  `sagemaker`).
Local uses `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` in the
API process. SageMaker invokes a deployed endpoint via boto3 with tenacity
retry. Both code paths are import-clean and type-checked in CI; only the local
path is invoked by tests.

### Consequences

- Reviewers can run the system without AWS.
- The SageMaker code is real (not stubbed) but exercised separately.

---

## ADR-001: Tenant isolation enforced at the query layer

**Date:** 2026-04-25
**Status:** Accepted

### Context

Route-layer authorization is necessary but not sufficient: a service mistake
or a future endpoint forgetting a check could leak cross-tenant data.

### Decision

Every tenant-scoped service method takes `organization_id` as a required
parameter. Every retrieval / listing query passes through
`apply_tenant_filter` (or an explicit equivalent `WHERE organization_id = :id`
predicate). A unit test (Stage 3) uses AST inspection to fail the build if
any service method touching tenant tables omits `organization_id`.

The `chunks` table denormalizes `organization_id` so retrieval queries can
filter without a join.

### Consequences

- Slight write cost (one extra column on `chunks`).
- Read path is simple, fast, and auditable.
- Any future endpoint inherits tenant isolation by default just by following
  the service-method convention.
