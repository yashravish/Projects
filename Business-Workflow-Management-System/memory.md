# Memory

Use this file as the persistent project memory while building.

The project is a full-stack Business Workflow Management System designed to fill resume gaps for a Programmer/Analyst role.

The job description values practical business application development more than advanced AI. Therefore, prioritize clean CRUD functionality, OOP design, relational database modeling, SQL reporting, testing, and debugging.

## Non-Negotiable Requirements

Always preserve these project goals:

1. The app must demonstrate Object-Oriented Programming.
2. The app must demonstrate HTML, CSS, and JavaScript through a React frontend.
3. The app must demonstrate SQL and relational database design through PostgreSQL.
4. The app must demonstrate problem solving and troubleshooting through validation, error handling, audit logs, and tests.
5. The app must simulate real business workflows.
6. The app must be simple enough to understand, but complete enough to be resume-worthy.
7. The app must include backend, frontend, database, Docker, tests, seed data, and README.
8. The app must not become an AI project.

## Architecture Decisions

Use this stack:

Frontend:
- React
- Vite
- TypeScript
- React Router
- Plain CSS or CSS modules
- Vitest
- React Testing Library
- Playwright

Backend:
- Python
- FastAPI
- SQLAlchemy 2.0
- Alembic
- PostgreSQL
- Pydantic
- JWT auth
- Pytest

Infrastructure:
- Docker Compose
- Makefile
- .env.example files

## Backend Folder Structure

Use:

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
      user.py
      task.py
      workflow_event.py
      approval.py
      audit_log.py
    schemas/
      auth.py
      user.py
      task.py
      workflow_event.py
      approval.py
      audit_log.py
      report.py
    repositories/
      user_repository.py
      task_repository.py
      audit_log_repository.py
    services/
      auth_service.py
      task_service.py
      workflow_service.py
      approval_service.py
      audit_log_service.py
      report_service.py
    api/
      deps.py
      routes/
        auth.py
        tasks.py
        reports.py
        audit_logs.py
    tests/

## Frontend Folder Structure

Use:

frontend/
  src/
    main.tsx
    App.tsx
    api/
      client.ts
      auth.ts
      tasks.ts
      reports.ts
      auditLogs.ts
    components/
      Layout.tsx
      NavBar.tsx
      ProtectedRoute.tsx
      StatusBadge.tsx
      PriorityBadge.tsx
      LoadingState.tsx
      ErrorMessage.tsx
    pages/
      LoginPage.tsx
      DashboardPage.tsx
      TaskListPage.tsx
      CreateTaskPage.tsx
      TaskDetailPage.tsx
      EditTaskPage.tsx
      ReportsPage.tsx
      AuditLogPage.tsx
    types/
      auth.ts
      task.ts
      report.ts
    tests/
    styles/
      global.css

## Business Rules to Remember

Statuses:

- pending
- in_progress
- submitted
- approved
- rejected
- completed

Valid transitions:

- pending → in_progress
- pending → submitted
- in_progress → submitted
- submitted → approved
- submitted → rejected
- rejected → in_progress
- rejected → submitted
- approved → completed

Invalid transitions must return HTTP 400 with a useful error message.

Role permissions:

- Analyst can create tasks.
- Analyst can submit tasks.
- Analyst cannot approve or reject tasks.
- Manager can approve or reject submitted tasks.
- Manager can complete approved tasks.
- Completed tasks cannot be edited.
- Approved or completed tasks cannot be deleted.

Audit rules:

Every important action creates an audit log:

- login
- create_task
- update_task
- delete_task
- submit_task
- approve_task
- reject_task
- complete_task

Workflow rules:

Every status change creates a workflow event.

Approval rules:

Every approve or reject decision creates an approval record.

## Seed Data Memory

Required login credentials:

Manager:
- manager@example.com
- password123

Analyst:
- analyst@example.com
- password123

Seed at least:
- 2 managers
- 3 analysts
- 12 tasks
- workflow events
- approvals
- audit logs

## Testing Memory

Backend tests must cover:

- Login works
- Protected route rejects unauthenticated users
- Task CRUD works
- Invalid transitions fail
- Manager approval works
- Analyst approval fails
- Audit logs are created
- Workflow events are created
- Reports aggregate data correctly
- CSV export works

Frontend tests must cover:

- Login renders
- Login error displays
- Dashboard summary renders
- Task form validation works
- Task list displays data
- Manager-only buttons render conditionally
- Error states render

E2E tests must cover:

- Analyst login
- Create task
- Submit task
- Manager login
- Approve task
- Dashboard update
- Reports load
- CSV export action

## Style Memory

Code style:
- Clean names
- Strong typing
- Small functions
- Service-layer business logic
- Route handlers should be thin
- Avoid duplication
- Avoid unnecessary abstraction
- Avoid placeholder comments
- Avoid TODOs

UI style:
- Professional
- Clean
- Business-app feel
- Responsive
- Accessible
- Simple CSS

Documentation style:
- Clear
- Resume-oriented
- Practical
- Explain how the project maps to the job description

## README Resume Bullets

Include these exactly or with small improvements:

- Designed and developed a full-stack business workflow application using React, FastAPI, PostgreSQL, and SQLAlchemy to manage task assignment, approvals, audit logs, and operational reporting.
- Applied object-oriented programming principles through service-layer classes for authentication, workflow transitions, approvals, reporting, and audit logging.
- Built normalized relational database schemas with foreign keys, workflow history tables, approval records, and SQL aggregation queries for business reporting.
- Implemented end-to-end testing with Pytest, Vitest, React Testing Library, and Playwright to validate CRUD operations, business rules, role permissions, and dashboard workflows.
- Debugged and hardened application behavior with structured validation, error handling, audit logs, and test coverage across frontend, backend, and database layers.

## Final Output Standard

The project is not done unless:

- Docker Compose runs the full app.
- Backend starts successfully.
- Frontend starts successfully.
- Database migrations exist.
- Seed data works.
- Backend tests exist and are meaningful.
- Frontend tests exist and are meaningful.
- Playwright tests exist and reflect real workflows.
- README is complete.
- Resume bullets are included.