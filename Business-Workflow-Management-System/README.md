# Business Workflow Management System

An ERP-lite internal workflow application that lets analysts create and submit
business tasks, lets managers approve, reject, and complete them, and gives both
roles a dashboard, audit trail, and operational reports.

This project is intentionally **not** an AI app. It is a clean, business-focused
full-stack CRUD system designed to demonstrate the practical software engineering
fundamentals required for a Programmer/Analyst role: object-oriented design,
HTML/CSS/JavaScript, SQL and relational modelling, debugging, validation, and
end-to-end testing.

---

## Why this project exists

The system models a real internal business process used by analyst teams in
finance, operations, and IT. It demonstrates the ability to:

- Translate business requirements into a working application
- Design and develop a CRUD application backed by a relational database
- Use object-oriented programming to organize business logic
- Build a clean, accessible web UI for analysts and managers
- Debug and harden the application with validation, error handling, and audit logs
- Cover the application end-to-end with automated tests

---

## Features

- **Authentication** with JWT tokens and role-based access (`analyst`, `manager`)
- **Tasks** with full CRUD: title, description, priority, status, assignee
- **Workflow state machine** for valid transitions:
  - `pending → in_progress`, `pending → submitted`
  - `in_progress → submitted`
  - `submitted → approved`, `submitted → rejected`
  - `rejected → in_progress`, `rejected → submitted`
  - `approved → completed`
- **Approvals** recorded for every approve / reject decision
- **Workflow events** that record every status change with actor and timestamp
- **Audit logs** for every important business action
- **Dashboard** showing summary cards, my work, and recent activity
- **Reports** with status, priority, user-workload, and approval aggregations
- **CSV export** of all tasks
- **Seed data** with 5 users (2 managers + 3 analysts) and 12 realistic tasks

---

## Tech stack

| Layer        | Technology |
| ------------ | ---------- |
| Frontend     | React, Vite, TypeScript, React Router, plain CSS |
| Backend      | Python, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Database     | PostgreSQL 16 |
| Auth         | JWT (HS256, python-jose), bcrypt password hashing |
| Tests        | Pytest, Vitest, React Testing Library, Playwright |
| Infra        | Docker Compose, Makefile, .env files |

---

## Architecture

```
+------------+        HTTPS/JSON       +-----------------+        SQL        +-------------+
|  React UI  |  <------------------->  |  FastAPI API     |  <----------->   | PostgreSQL  |
| (Vite/TSX) |                         | services / OOP   |                  |             |
+------------+                         +-----------------+                   +-------------+
                                              |
                                              v
                              Audit logs + Workflow events table
```

The backend follows a clean layered architecture:

```
backend/app/
  core/        # config + security primitives
  db/          # SQLAlchemy session, declarative base, seed script
  models/      # ORM tables: User, Task, WorkflowEvent, Approval, AuditLog
  schemas/    # Pydantic request/response models
  repositories/# Thin data-access objects
  services/    # OOP business logic (AuthService, TaskService, ...)
  api/routes/  # FastAPI route handlers (thin – they call services)
  tests/       # Pytest suite
```

Route handlers are intentionally thin. All business rules (status transitions,
permission checks, audit logging) live in the **service classes**:

- `AuthService` – login, JWT issuance, current-user resolution
- `TaskService` – task CRUD, lifecycle, role checks
- `WorkflowService` – status transition validation + workflow events
- `ApprovalService` – approve / reject decisions
- `AuditLogService` – central audit log writer
- `ReportService` – SQL aggregation queries + CSV export

---

## Database schema

The database is a normalized relational schema with foreign keys, indexes,
timestamps, and PostgreSQL enum-like constraints.

```
users (id PK, name, email UNIQUE, hashed_password, role ENUM, created_at, updated_at)

tasks (
  id PK, title, description, status ENUM, priority ENUM,
  assigned_to_user_id FK -> users.id,
  created_by_user_id FK -> users.id,
  created_at, updated_at
)

workflow_events (
  id PK, task_id FK -> tasks.id (CASCADE),
  from_status ENUM NULL, to_status ENUM,
  changed_by_user_id FK -> users.id,
  note, created_at
)

approvals (
  id PK, task_id FK -> tasks.id (CASCADE),
  approved_by_user_id FK -> users.id,
  decision ENUM (approved | rejected),
  comment, created_at
)

audit_logs (
  id PK, user_id FK -> users.id NULL,
  action ENUM, entity_type, entity_id,
  details, created_at
)
```

Reporting endpoints use real SQL aggregation (`GROUP BY status`, `GROUP BY
priority`, per-user breakdowns, approval decisions).

---

## Business rules

- Analysts can create, edit, submit, and delete tasks they created (when not
  approved/completed). They cannot approve, reject, or complete tasks.
- Managers can view all tasks, edit any non-completed task, approve or reject
  submitted tasks, and complete approved tasks.
- Completed tasks are immutable.
- Approved or completed tasks cannot be deleted.
- Every status change creates a workflow event.
- Every approve/reject decision creates an approval record.
- Every important business action (login, create/update/delete, submit, approve,
  reject, complete) creates an audit log.
- Invalid status transitions return a clear `400 Bad Request`.

---

## API overview

```
POST   /auth/login                    -> { access_token, token_type, user }
GET    /auth/me                       -> Current user
GET    /users                          -> List users (used for assignee dropdown)

GET    /tasks                          -> List tasks (filters: status, priority, assignee)
POST   /tasks                          -> Create task
GET    /tasks/{id}                     -> Read task
PUT    /tasks/{id}                     -> Update task
DELETE /tasks/{id}                     -> Delete task
POST   /tasks/{id}/submit              -> Move to submitted
POST   /tasks/{id}/approve             -> Manager-only approve
POST   /tasks/{id}/reject              -> Manager-only reject
POST   /tasks/{id}/complete            -> Manager-only complete
GET    /tasks/{id}/workflow-events     -> Workflow history for a task

GET    /audit-logs                     -> Audit log feed (filters: action, limit)

GET    /reports/tasks-by-status        -> Aggregated counts
GET    /reports/tasks-by-priority      -> Aggregated counts
GET    /reports/user-workload          -> Per-user breakdown by status
GET    /reports/approval-summary       -> Approve/reject decision counts
GET    /reports/export/tasks.csv       -> CSV export of all tasks
```

OpenAPI/Swagger docs are available at <http://localhost:8000/docs>.

---

## Setup

### Prerequisites

- Docker & Docker Compose
- (Optional) Node 20+ and Python 3.11+ for running tests outside Docker

### Run with Docker (recommended)

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
make up
```

This will:

1. Start PostgreSQL on port 5432
2. Start the FastAPI backend on port 8000 (runs Alembic migrations + seed on boot)
3. Start the Vite frontend on port 5173

Open the app at **<http://localhost:5173>** and sign in with one of the seeded
accounts.

### Demo credentials

| Role     | Email                        | Password    |
| -------- | ---------------------------- | ----------- |
| Manager  | manager@example.com          | password123 |
| Manager  | marcus.manager@example.com   | password123 |
| Analyst  | analyst@example.com          | password123 |
| Analyst  | brian.analyst@example.com    | password123 |
| Analyst  | carla.analyst@example.com    | password123 |

### Running locally without Docker

Backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://workflow:workflow@localhost:5432/workflow
export JWT_SECRET_KEY=dev-secret
alembic upgrade head
python -m app.db.seed
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## Testing

### Backend (Pytest)

```bash
cd backend
pip install -r requirements.txt
pytest
```

The Pytest suite uses an in-memory SQLite database via dependency overrides and
covers:

- Auth login + JWT issuance
- Protected route rejection for unauthenticated requests
- Task CRUD
- All workflow status transitions, including invalid transitions returning 400
- Manager approval / rejection / completion paths
- Analyst permission failures (cannot approve, reject, complete)
- Audit log creation for every important action
- Workflow event creation for every status change
- Report aggregations (status, priority, user workload, approval summary)
- CSV export endpoint

### Frontend (Vitest + React Testing Library)

```bash
cd frontend
npm install
npm run test
```

Covers:

- Login form rendering, validation errors, and API error handling
- Dashboard summary cards rendering with mocked report data
- Task list rendering rows from the API
- Create task form validation and submission
- Manager-only buttons appearing only for manager users
- Status / priority badge rendering
- API error message component

### End-to-end (Playwright)

```bash
cd frontend
npm install
npx playwright install --with-deps
make up         # in another terminal, ensure stack is up
npm run test:e2e
```

Covers the full business flow:

1. Analyst logs in and creates a task
2. Analyst submits the task
3. Manager logs in
4. Manager approves a submitted task
5. Reports page loads and the CSV export button is present (and triggers a
   download)
6. Audit log page renders entries

### Makefile shortcuts

```bash
make test            # backend + frontend tests inside containers
make test-backend
make test-frontend
make test-e2e        # against the running compose stack
make seed            # re-seed demo data
```

---

## Screenshots

> Screenshots placeholder. Capture the dashboard, task detail, reports, and
> audit log pages once the stack is running locally.

- `screenshots/dashboard.png`
- `screenshots/task-detail.png`
- `screenshots/reports.png`
- `screenshots/audit-log.png`

---

## Programmer/Analyst skill mapping

| Skill from the job description | Where it lives in this project |
| --- | --- |
| **Object-oriented programming** | `backend/app/services/*` – every business rule lives in a service class (`AuthService`, `TaskService`, `WorkflowService`, `ApprovalService`, `AuditLogService`, `ReportService`). Route handlers are thin and delegate to these classes. |
| **HTML / CSS / JavaScript** | `frontend/src/**` – React components rendering semantic HTML (`<main>`, `<nav>`, `<form>`, accessible labels, alert/status roles) styled with hand-written, responsive CSS in `styles/global.css`. |
| **SQL and relational databases** | Normalized PostgreSQL schema (`users`, `tasks`, `workflow_events`, `approvals`, `audit_logs`) with foreign keys and indexes. Reporting endpoints use SQLAlchemy `GROUP BY` aggregations (`ReportService`). Alembic migrations live in `backend/alembic/versions/`. |
| **CRUD application development** | `POST/GET/PUT/DELETE /tasks` plus workflow actions (`/submit`, `/approve`, `/reject`, `/complete`) wired end-to-end through services, routes, and React pages. |
| **Business requirement translation** | The product instruction files (`agents.md`, `context.md`, `memory.md`, `skills.md`) describe a real internal workflow and are implemented exactly: roles, transitions, permissions, audits, reports. |
| **Debugging, validation, troubleshooting** | Pydantic input validation, role-based dependency checks, central transition validator with clear 400 errors, structured logging, exhaustive audit trail, and tests that exercise the failure modes. |
| **Automated testing** | Pytest (backend), Vitest + React Testing Library (frontend), Playwright (E2E) covering CRUD, role permissions, business rules, and dashboards. |
| **Ability to learn enterprise / proprietary environments** | The codebase mirrors enterprise patterns: service layer, repositories, JWT auth, ORM, migrations, audit trail, role-based access. The structure is designed to make onboarding to a similar enterprise stack (e.g., RPG/iSeries, internal Java/.NET ERPs) straightforward. |

---

## Resume bullets

- Designed and developed a full-stack business workflow application using React, FastAPI, PostgreSQL, and SQLAlchemy to manage task assignment, approvals, audit logs, and operational reporting.
- Applied object-oriented programming principles through service-layer classes for authentication, workflow transitions, approvals, reporting, and audit logging.
- Built normalized relational database schemas with foreign keys, workflow history tables, approval records, and SQL aggregation queries for business reporting.
- Implemented end-to-end testing with Pytest, Vitest, React Testing Library, and Playwright to validate CRUD operations, business rules, role permissions, and dashboard workflows.
- Debugged and hardened application behavior with structured validation, error handling, audit logs, and test coverage across frontend, backend, and database layers.
