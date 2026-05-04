# Skills

Use these skills while building the Business Workflow Management System.

## Skill 1: Requirements Translation

Translate business requirements into application features.

The job description says:
- Design, develop, test, and debug applications.
- Satisfy requirements of various business functions.
- Work with development team and management.
- Identify and specify business requirements and processes.

In this project, represent those requirements as:

- Task assignment
- Workflow status tracking
- Manager approvals
- Analyst submissions
- Audit logs
- Business reports
- CSV export
- Role-based access

When writing code or documentation, emphasize that the system models a business process.

## Skill 2: Object-Oriented Programming

Use OOP through backend service classes.

Required service classes:

- AuthService
- TaskService
- WorkflowService
- ApprovalService
- AuditLogService
- ReportService

Each class should have clear responsibility.

Examples:

TaskService:
- create_task
- update_task
- delete_task
- submit_task
- complete_task
- validate_transition

WorkflowService:
- create_workflow_event
- get_task_history

ApprovalService:
- approve_task
- reject_task

AuditLogService:
- log_action
- list_logs

ReportService:
- tasks_by_status
- tasks_by_priority
- user_workload
- approval_summary
- export_tasks_csv

Keep route handlers thin. They should call services.

## Skill 3: SQL and Relational Database Design

Use PostgreSQL and SQLAlchemy to model normalized relational data.

Required tables:

- users
- tasks
- workflow_events
- approvals
- audit_logs

Use relationships, foreign keys, indexes, and timestamps.

Reporting endpoints should use actual SQL aggregation or SQLAlchemy aggregation.

Examples:
- Count tasks grouped by status.
- Count tasks grouped by priority.
- Count tasks assigned to each user.
- Count approvals grouped by decision.

The database should clearly demonstrate relational thinking.

## Skill 4: CRUD Development

Implement complete CRUD for tasks.

Create:
- POST /tasks

Read:
- GET /tasks
- GET /tasks/{task_id}

Update:
- PUT /tasks/{task_id}

Delete:
- DELETE /tasks/{task_id}

Also implement business workflow actions:

- POST /tasks/{task_id}/submit
- POST /tasks/{task_id}/approve
- POST /tasks/{task_id}/reject
- POST /tasks/{task_id}/complete

Frontend must provide UI for these actions.

## Skill 5: Authentication and Authorization

Implement JWT-based authentication.

Seed users:

- manager@example.com / password123
- analyst@example.com / password123

Rules:

- Unauthenticated users cannot access protected routes.
- Analysts cannot approve or reject.
- Managers can approve and reject submitted tasks.
- Users should only see buttons they are allowed to use.
- Backend must still enforce permissions even if frontend hides buttons.

## Skill 6: Workflow State Machine

Implement a clean status transition system.

Allowed transitions:

- pending → in_progress
- pending → submitted
- in_progress → submitted
- submitted → approved
- submitted → rejected
- rejected → in_progress
- rejected → submitted
- approved → completed

Disallowed transitions return HTTP 400.

Completed tasks are final and cannot be edited.

Use a central transition validator in TaskService or WorkflowService.

## Skill 7: Audit Logging

Every meaningful action must create an audit log.

Actions:

- login
- create_task
- update_task
- delete_task
- submit_task
- approve_task
- reject_task
- complete_task

Audit logs should include:
- user_id
- action
- entity_type
- entity_id
- details
- created_at

This demonstrates troubleshooting, traceability, and enterprise-style accountability.

## Skill 8: Error Handling and Troubleshooting

Implement clear error handling.

Backend:
- Use HTTPException with clear messages.
- Validate input with Pydantic.
- Validate permissions.
- Validate status transitions.
- Handle missing resources with 404.
- Handle invalid actions with 400.
- Handle unauthorized access with 401 or 403.

Frontend:
- Show loading states.
- Show error messages.
- Prevent invalid form submissions.
- Handle failed API requests.
- Keep the UI usable when errors happen.

## Skill 9: Backend Testing

Use Pytest.

Tests must be meaningful and should test actual business logic.

Required backend test areas:

- Auth login
- Protected routes
- Task CRUD
- Workflow status transitions
- Role permissions
- Manager approval
- Analyst approval failure
- Audit log creation
- Workflow event creation
- Report aggregation
- CSV export

Use test database setup that works reliably.

## Skill 10: Frontend Testing

Use Vitest and React Testing Library.

Required frontend test areas:

- Login form renders
- Login error renders
- Dashboard summary cards render
- Task form validates required fields
- Task list displays task rows
- Role-based buttons render correctly
- API error messages display

Tests should focus on user-visible behavior.

## Skill 11: End-to-End Testing

Use Playwright.

Test the most important full workflow:

1. Analyst logs in.
2. Analyst creates a task.
3. Analyst submits task.
4. Manager logs in.
5. Manager approves task.
6. Dashboard reflects changes.
7. Reports page loads.
8. CSV export action works.

The E2E test should prove the app works as a business workflow system.

## Skill 12: Docker and Local Development

Create Docker-based local setup.

Required:
- docker-compose.yml
- backend/Dockerfile
- frontend/Dockerfile
- PostgreSQL service
- backend service
- frontend service

Make sure environment variables are clear.

Use .env.example files.

Expose:
- frontend on 5173 or 3000
- backend on 8000
- database on 5432

## Skill 13: Makefile Automation

Create Makefile commands:

- make up
- make down
- make test
- make test-backend
- make test-frontend
- make test-e2e
- make seed

Commands must match the actual project structure.

## Skill 14: Seed Data

Create a seed script that inserts useful demo data.

Seed:
- 2 managers
- 3 analysts
- 12 tasks
- Multiple statuses
- Multiple priorities
- Workflow events
- Approvals
- Audit logs

Seed script must be safe to rerun if possible.

## Skill 15: Documentation

README must include:

- Project name
- Overview
- Why it exists
- Features
- Tech stack
- Architecture
- Database schema
- Business rules
- Setup instructions
- Testing instructions
- Demo login credentials
- Programmer/Analyst skill mapping
- Resume bullets

The documentation should make it obvious why this project is useful for the target role.

## Skill 16: Resume Alignment

The project must support these resume bullets:

- Designed and developed a full-stack business workflow application using React, FastAPI, PostgreSQL, and SQLAlchemy to manage task assignment, approvals, audit logs, and operational reporting.
- Applied object-oriented programming principles through service-layer classes for authentication, workflow transitions, approvals, reporting, and audit logging.
- Built normalized relational database schemas with foreign keys, workflow history tables, approval records, and SQL aggregation queries for business reporting.
- Implemented end-to-end testing with Pytest, Vitest, React Testing Library, and Playwright to validate CRUD operations, business rules, role permissions, and dashboard workflows.
- Debugged and hardened application behavior with structured validation, error handling, audit logs, and test coverage across frontend, backend, and database layers.

## Skill 17: Final Verification

Before stopping, verify that:

- All expected files exist.
- Backend has no obvious broken imports.
- Frontend has no obvious broken imports.
- API route paths match frontend calls.
- Environment variable names match Docker Compose.
- Database models match schemas and migrations.
- Tests target implemented features.
- README commands match actual commands.
- No placeholder code remains.
- No TODO comments remain.