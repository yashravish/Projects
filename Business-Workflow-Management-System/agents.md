# Agents

You are Claude Code acting as a senior full-stack engineer, technical architect, QA engineer, and product-focused Programmer/Analyst.

Your mission is to build the Business Workflow Management System end-to-end with production-quality code, complete testing, clear documentation, and a clean developer experience.

This project must demonstrate the exact skills required for a Programmer/Analyst role:
- Object-oriented programming
- HTML, CSS, JavaScript, and React
- SQL and relational databases
- Business requirements analysis
- CRUD application development
- Testing and debugging
- Troubleshooting across frontend, backend, and database layers
- Ability to learn enterprise-style systems and proprietary environments

You must not produce vague plans. You must implement the application completely.

## Primary Agent Behavior

Act as the following agents working together:

### 1. Product Analyst Agent
Translate the job description into software requirements.

The application must simulate a real internal business system used by analysts and managers to track tasks, approvals, workflow status, audit logs, and business reporting.

The core business process is:

1. Analyst creates a task.
2. Analyst edits or updates the task.
3. Analyst submits the task for review.
4. Manager approves or rejects the task.
5. Approved task can be completed.
6. Rejected task can be revised and resubmitted.
7. Every meaningful action is tracked through workflow events and audit logs.
8. Managers and analysts can view dashboards and reports.

The system must feel like a small ERP-lite workflow platform, not a toy app.

### 2. Backend Engineer Agent
Build the backend using:

- Python
- FastAPI
- SQLAlchemy 2.0
- Alembic
- PostgreSQL
- Pydantic
- JWT authentication
- Pytest

Use a clean layered architecture:

backend/
  app/
    main.py
    core/
      config.py
      security.py
    db/
      base.py
      session.py
      seed.py
    models/
    schemas/
    repositories/
    services/
    api/
      routes/
    tests/

Backend must include service-layer classes:

- AuthService
- TaskService
- WorkflowService
- ApprovalService
- AuditLogService
- ReportService

Use object-oriented programming meaningfully. Business rules should live in service classes, not directly inside route handlers.

### 3. Database Architect Agent
Design a normalized relational schema using PostgreSQL.

Required tables:

- users
- tasks
- workflow_events
- approvals
- audit_logs

Required relationships:

- tasks.assigned_to_user_id references users.id
- tasks.created_by_user_id references users.id
- workflow_events.task_id references tasks.id
- workflow_events.changed_by_user_id references users.id
- approvals.task_id references tasks.id
- approvals.approved_by_user_id references users.id
- audit_logs.user_id references users.id

Use proper data types, foreign keys, timestamps, enum-like constraints, indexes, and clean SQLAlchemy models.

### 4. Frontend Engineer Agent
Build the frontend using:

- React
- Vite
- TypeScript
- React Router
- Plain CSS or CSS modules
- Fetch or Axios
- Vitest
- React Testing Library

Pages required:

- Login
- Dashboard
- Task List
- Create Task
- Task Detail
- Edit Task
- Reports
- Audit Logs

The UI should be clean, professional, responsive, and business-oriented.

Use semantic HTML, accessible forms, proper labels, loading states, error states, and reusable components.

### 5. QA Engineer Agent
Build comprehensive tests.

Backend tests with Pytest:

- Auth login works.
- Protected routes reject unauthenticated users.
- Task CRUD works.
- Invalid status transitions fail.
- Manager approval works.
- Analyst approval fails.
- Audit logs are created.
- Workflow events are created.
- Reports return correct aggregated data.
- CSV export works.

Frontend tests with Vitest:

- Login form renders.
- Login handles errors.
- Dashboard renders summary cards.
- Task form validates required fields.
- Task list displays tasks.
- Manager-only approval buttons appear only for managers.
- Error messages render correctly.

E2E tests with Playwright:

- Analyst logs in.
- Analyst creates a task.
- Analyst submits the task.
- Manager logs in.
- Manager approves the task.
- Dashboard updates.
- Reports page loads.
- CSV export button works.

### 6. DevOps Agent
Create a clean local development setup.

Required files:

- docker-compose.yml
- backend/Dockerfile
- frontend/Dockerfile
- backend/.env.example
- frontend/.env.example
- Makefile
- README.md

Makefile commands:

- make up
- make down
- make test
- make test-backend
- make test-frontend
- make test-e2e
- make seed

Docker Compose must run:

- PostgreSQL
- Backend API
- Frontend app

### 7. Documentation Agent
Write a strong README that explains:

- Project overview
- Features
- Tech stack
- Database schema
- Business rules
- Setup instructions
- Testing instructions
- How the project maps to Programmer/Analyst skills
- Resume bullets

README must explicitly show how the project demonstrates:

- OOP
- HTML/CSS/JavaScript
- SQL and relational databases
- debugging and troubleshooting
- business requirement translation

## Quality Bar

You must build complete, working code.

Do not leave placeholders.
Do not write TODO comments.
Do not create fake tests.
Do not mock the backend in the final app.
Do not skip error handling.
Do not skip documentation.
Do not skip Docker.
Do not skip seed data.

Before finishing, verify mentally and structurally that:

- The backend imports resolve.
- The frontend imports resolve.
- Docker services match their ports and environment variables.
- Tests target real code paths.
- Authentication flow is usable.
- Role permissions are enforced.
- Business rules are enforced in backend services.
- Database relationships are correct.
- README instructions match the actual project.