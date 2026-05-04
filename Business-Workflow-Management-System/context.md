# Context

This project is being built specifically to fill resume gaps for a Programmer/Analyst job.

The target job description emphasizes:

- Designing, developing, testing, and debugging applications for business functions
- Working with development teams and management to identify requirements and processes
- Object-oriented programming
- HTML, CSS, and JavaScript
- SQL and relational databases
- Problem solving and troubleshooting
- RPG ILE as a plus
- IBM Power Systems iSeries as a plus
- Strong communication and interpersonal skills
- Willingness to learn proprietary development environments

The user already has strong experience in AI systems, backend development, FastAPI, React, testing, and technical projects. However, the user’s resume currently leans heavily toward AI, RAG, agentic systems, and ML infrastructure.

This project must reposition the user as someone who can build practical business applications.

The project should therefore avoid being an AI-heavy application. The value is not machine learning. The value is business software fundamentals.

## Project Name

Business Workflow Management System

Alternative description:

An ERP-lite internal workflow application for managing tasks, approvals, audit logs, and business reporting.

## Core Positioning

This project should show that the user can take business requirements and turn them into working software.

The project must clearly demonstrate:

1. Business requirement translation
2. CRUD application development
3. Object-oriented design
4. Relational database modeling
5. SQL reporting
6. Role-based workflow logic
7. Debugging and troubleshooting
8. Automated testing
9. Clean documentation
10. Enterprise-style application structure

## Target Users

There are two user roles:

### Analyst
Analysts create, update, and submit tasks.

Analysts can:
- Log in
- View dashboard
- Create tasks
- Edit their own non-completed tasks
- Submit tasks for manager approval
- View workflow history
- View reports
- View audit logs

Analysts cannot:
- Approve tasks
- Reject tasks
- Edit completed tasks
- Delete approved or completed tasks

### Manager
Managers review submitted tasks.

Managers can:
- Log in
- View dashboard
- View all tasks
- Approve submitted tasks
- Reject submitted tasks
- Complete approved tasks
- View reports
- View audit logs

Managers cannot:
- Approve tasks that are not submitted
- Complete tasks that are not approved
- Bypass workflow rules

## Required Business Rules

Statuses:

- pending
- in_progress
- submitted
- approved
- rejected
- completed

Priorities:

- low
- medium
- high

Valid transitions:

- pending → in_progress
- pending → submitted
- in_progress → submitted
- submitted → approved
- submitted → rejected
- rejected → in_progress
- rejected → submitted
- approved → completed

Invalid examples:

- pending → completed
- in_progress → completed
- rejected → completed
- completed → any other status
- approved → rejected
- submitted → completed without approval

Role rules:

- Analysts can create and submit tasks.
- Managers can approve or reject submitted tasks.
- Only approved tasks can be completed.
- Completed tasks cannot be edited.
- Deleted tasks must not be approved or completed.
- Every important action creates an audit log.
- Every status change creates a workflow event.
- Every approval or rejection creates an approval record.

## Required Data Model

Tables:

### users
Fields:
- id
- name
- email
- hashed_password
- role
- created_at
- updated_at

### tasks
Fields:
- id
- title
- description
- status
- priority
- assigned_to_user_id
- created_by_user_id
- created_at
- updated_at

### workflow_events
Fields:
- id
- task_id
- from_status
- to_status
- changed_by_user_id
- note
- created_at

### approvals
Fields:
- id
- task_id
- approved_by_user_id
- decision
- comment
- created_at

Decision values:
- approved
- rejected

### audit_logs
Fields:
- id
- user_id
- action
- entity_type
- entity_id
- details
- created_at

## Required Seed Users

Seed these users:

1. Manager
Email: manager@example.com
Password: password123
Role: manager

2. Analyst
Email: analyst@example.com
Password: password123
Role: analyst

Also seed:
- 2 managers total
- 3 analysts total
- 12 tasks across different statuses
- sample workflow events
- sample approvals
- sample audit logs

## Required Backend API

Authentication:
- POST /auth/login
- GET /auth/me

Tasks:
- GET /tasks
- POST /tasks
- GET /tasks/{task_id}
- PUT /tasks/{task_id}
- DELETE /tasks/{task_id}
- POST /tasks/{task_id}/submit
- POST /tasks/{task_id}/approve
- POST /tasks/{task_id}/reject
- POST /tasks/{task_id}/complete

Workflow:
- GET /tasks/{task_id}/workflow-events

Audit:
- GET /audit-logs

Reports:
- GET /reports/tasks-by-status
- GET /reports/tasks-by-priority
- GET /reports/user-workload
- GET /reports/approval-summary
- GET /reports/export/tasks.csv

## Required Frontend Pages

### Login Page
- Email input
- Password input
- Submit button
- Error display
- Demo credentials visible on page

### Dashboard Page
Show:
- Total tasks
- Tasks by status
- Tasks by priority
- My assigned tasks
- Recent workflow events
- Recent audit logs

### Task List Page
Show:
- Table/list of tasks
- Status badge
- Priority badge
- Assigned user
- Created date
- Link to detail page
- Filters by status and priority

### Create Task Page
Form:
- Title
- Description
- Priority
- Assigned user
- Submit

### Task Detail Page
Show:
- Task details
- Status
- Priority
- Assigned user
- Workflow history
- Action buttons based on role and status

### Edit Task Page
Form:
- Title
- Description
- Priority
- Assigned user

### Reports Page
Show:
- Tasks by status
- Tasks by priority
- User workload
- Approval summary
- CSV export button

### Audit Log Page
Show:
- Recent actions
- User
- Entity type
- Entity ID
- Details
- Timestamp

## Design Requirements

The UI should be professional, simple, and clean.

Use:
- Semantic HTML
- Accessible labels
- Responsive layout
- Clear navigation
- Cards
- Tables
- Badges
- Forms
- Error messages
- Loading states

Avoid:
- Overly fancy animations
- Heavy UI libraries
- Unnecessary complexity
- AI branding
- Toy-looking design

## Resume Positioning

The README must include these resume bullets:

- Designed and developed a full-stack business workflow application using React, FastAPI, PostgreSQL, and SQLAlchemy to manage task assignment, approvals, audit logs, and operational reporting.
- Applied object-oriented programming principles through service-layer classes for authentication, workflow transitions, approvals, reporting, and audit logging.
- Built normalized relational database schemas with foreign keys, workflow history tables, approval records, and SQL aggregation queries for business reporting.
- Implemented end-to-end testing with Pytest, Vitest, React Testing Library, and Playwright to validate CRUD operations, business rules, role permissions, and dashboard workflows.
- Debugged and hardened application behavior with structured validation, error handling, audit logs, and test coverage across frontend, backend, and database layers.

## Final Goal

The finished project should be strong enough to put on a resume for a Programmer/Analyst role.

It should make the user look like someone who can:

- Learn a proprietary business environment
- Understand business processes
- Build reliable internal software
- Work with relational data
- Debug application issues
- Communicate technical work clearly