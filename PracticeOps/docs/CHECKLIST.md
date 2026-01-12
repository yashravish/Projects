# PracticeOps MVP — Milestone Checklist

## Milestone 0: Repo + CI Foundation
- [x] Monorepo structure with `apps/api` and `apps/web`
- [x] Docker Compose with Postgres and API services
- [x] FastAPI with `/health` endpoint and `/openapi.json` exposed
- [x] SQLAlchemy 2.0 async setup
- [x] Alembic initialized (no domain migrations)
- [x] Pydantic v2 configuration
- [x] pytest with httpx async test client
- [x] Ruff + Black + Mypy configured
- [x] React + Vite + TypeScript
- [x] React Router configured
- [x] TanStack Query installed and wired
- [x] Vitest configured
- [x] Playwright configured
- [x] ESLint + Prettier configured
- [x] shadcn/ui initialized
- [x] CI workflow (lint, typecheck, tests)
- [x] README with one-command startup
- [x] CONTRIBUTING.md with test/CI instructions

## Milestone 1: Database Schema & Migrations
- [x] Domain enums (Role, AssignmentType, Priority, AssignmentScope, TicketCategory, TicketVisibility, TicketStatus, TicketActivityType)
- [x] SQLAlchemy 2.0 models for all 11 tables
- [x] users table with email, password_hash, display_name
- [x] teams table
- [x] team_memberships with composite unique (team_id, user_id)
- [x] rehearsal_cycles with composite unique (team_id, date)
- [x] assignments table with type, scope, section
- [x] practice_logs table
- [x] practice_log_assignments junction table
- [x] invites table with token and expiration
- [x] tickets table with visibility, status, priority, category
- [x] ticket_activity table for activity logging
- [x] notification_preferences with composite unique (team_id, user_id)
- [x] All indexes created explicitly (tickets, practice_logs, assignments, team_memberships)
- [x] Alembic migration with proper upgrade/downgrade
- [x] Seed script for initial data
- [x] docs/db.md with schema diagram and index rationale
- [x] Database constraint tests

## Milestone 2: Authentication & RBAC
- [x] JWT access token (15 min) + refresh token (30 days)
- [x] POST /auth/register endpoint
- [x] POST /auth/login endpoint
- [x] POST /auth/refresh endpoint
- [x] GET /me endpoint with primary team membership
- [x] Password hashing with passlib (bcrypt)
- [x] Standard error responses (UNAUTHORIZED, FORBIDDEN, CONFLICT, etc.)
- [x] require_auth() dependency
- [x] require_membership(team_id) dependency
- [x] require_role(team_id, roles) dependency
- [x] require_section_leader_of_section(team_id, section) dependency
- [x] Auth tests (register, login, refresh, /me, wrong password, no token)

## Milestone 3: Teams, Memberships & Invites
- [x] POST /teams - Team creation (creator becomes ADMIN)
- [x] GET /teams/{team_id}/members - List members with pagination (ADMIN, SECTION_LEADER)
- [x] PATCH /teams/{team_id}/members/{user_id} - Update membership (ADMIN only)
- [x] POST /teams/{team_id}/invites - Create invite with hashed tokens (ADMIN only)
- [x] GET /invites/{token} - Preview invite (public)
- [x] POST /invites/{token}/accept - Accept invite with pinned flow
- [x] Invite token hashing (raw token returned once, only hash stored)
- [x] Single-use tokens (used_at check)
- [x] Time-bound tokens (expires_at check)
- [x] ACCOUNT_EXISTS_LOGIN_REQUIRED flow for email-bound invites
- [x] RBAC enforcement via dependencies (not inline logic)
- [x] Pagination with deterministic ordering (created_at ASC, id ASC)
- [x] Comprehensive tests (18 tests covering all flows)

## Milestone 4: Rehearsal Cycles
- [x] POST /teams/{team_id}/cycles - Create cycle (ADMIN, SECTION_LEADER)
- [x] GET /teams/{team_id}/cycles - List cycles with pagination (any member)
- [x] GET /teams/{team_id}/cycles/active - Get active cycle (any member)
- [x] Active cycle logic (nearest upcoming >= today, else latest past, else null)
- [x] Deterministic pagination (date ASC/DESC + id tie-breaker)
- [x] Unique (team_id, date) constraint enforced
- [x] RBAC via dependencies (not inline logic)
- [x] Comprehensive tests (17 tests covering all flows)

## Milestone 5: Assignment System
- [x] POST /cycles/{cycle_id}/assignments - Create assignment (ADMIN, SECTION_LEADER with constraints)
- [x] GET /cycles/{cycle_id}/assignments - List with visibility rules and pagination
- [x] PATCH /assignments/{id} - Update assignment (creator or ADMIN)
- [x] DELETE /assignments/{id} - Delete assignment (ADMIN only)
- [x] RBAC: ADMIN creates TEAM/SECTION, SECTION_LEADER creates own section only
- [x] Visibility: Members see TEAM + their section's SECTION assignments
- [x] Server-set fields: team_id from cycle, created_by from auth, due_date = cycle.date
- [x] Sorting: priority DESC, due_at ASC, created_at DESC, id (deterministic)
- [x] Cursor-based pagination (no duplicates/missing across pages)
- [x] Scope validation (TEAM no section, SECTION requires section)
- [x] Schema migration for missing fields (priority, created_by, song_ref)
- [x] Comprehensive tests (23 tests covering all flows)

## Milestone 6: Practice Logging
- [x] POST /cycles/{cycle_id}/practice-logs - Create practice log
- [x] GET /cycles/{cycle_id}/practice-logs - List with me=true/false and pagination
- [x] PATCH /practice-logs/{id} - Update (owner only)
- [x] Assignment validation (must exist, belong to cycle, be visible to user)
- [x] Visibility rules: TEAM + user's SECTION assignments only
- [x] Privacy: me=true (own logs), me=false (SECTION_LEADER sees section, ADMIN sees all)
- [x] blocked_flag returns suggested_ticket object
- [x] Sorting: occurred_at DESC, id (deterministic)
- [x] Cursor-based pagination (no duplicates/missing across pages)
- [x] Join table: practice_log_assignments created properly
- [x] Schema migration for missing fields (rating_1_5, blocked_flag)
- [x] Comprehensive tests (17 tests covering all flows)

## Milestone 7: Ticket System
- [x] POST /cycles/{cycle_id}/tickets - Create ticket with visibility (PRIVATE/SECTION/TEAM)
- [x] GET /cycles/{cycle_id}/tickets - List with visibility enforcement and pagination
- [x] POST /tickets/{id}/claim - Claim claimable ticket (atomic, concurrency-safe)
- [x] PATCH /tickets/{id} - Update ticket (owner or leader in scope)
- [x] GET /tickets/{id}/activity - View ticket activity timeline
- [x] Visibility: PRIVATE (owner/creator), SECTION (section members), TEAM (all)
- [x] Claimable tickets: Leaders create, members claim (owner_id=null until claimed)
- [x] Activity logging: CREATED, CLAIMED events tracked
- [x] Double-claim returns CONFLICT (race-condition safe)
- [x] Sorting: priority DESC, due_at ASC, updated_at DESC, id (deterministic)
- [x] Cursor-based pagination (no duplicates/missing across pages)
- [x] Schema migration for missing fields (claimable, owner_id nullable)
- [x] Comprehensive tests (22 tests covering all flows)

## Milestone 8: Ticket Status Transitions + Verification
- [x] POST /tickets/{id}/transition - Status transitions with validation rules
- [x] POST /tickets/{id}/verify - Verification (ADMIN/SECTION_LEADER only)
- [x] Transition rules: OPEN <-> IN_PROGRESS <-> BLOCKED, IN_PROGRESS/BLOCKED -> RESOLVED
- [x] RESOLVED requires content (stored in resolved_note)
- [x] VERIFIED is terminal (cannot transition out)
- [x] Illegal transitions rejected (OPEN -> RESOLVED, OPEN -> VERIFIED)
- [x] Owner-only transition (creator or claimed owner)
- [x] Verify restricted to ADMIN or SECTION_LEADER in scope
- [x] Activity logging: STATUS_CHANGE and VERIFIED events with old_status/new_status
- [x] Atomic updates (ticket + activity in single transaction)
- [x] Schema migration for workflow fields (resolved_note, verified_by, verified_note)
- [x] Comprehensive tests (16 tests covering all flows)

## Milestone 9: Dashboards
- [x] GET /teams/{team_id}/dashboards/member - Personal progress dashboard
- [x] GET /teams/{team_id}/dashboards/leader - Team compliance dashboard
- [x] Member dashboard: assignments, tickets due, weekly summary, streak tracking
- [x] Leader dashboard: compliance metrics, risk summary, member drilldown
- [x] private_ticket_aggregates with strict privacy (no IDs/names, count >= 3 threshold)
- [x] Due bucket classification (overdue, due_today, due_this_week, future, no_due_date)
- [x] RBAC: Any member → member dashboard, ADMIN/SECTION_LEADER → leader dashboard
- [x] Section leader scope restrictions (only sees their section data)
- [x] Active cycle auto-selection
- [x] Countdown days calculation (positive/negative)
- [x] Comprehensive tests (15 tests covering all flows)

## Milestone 10a: Frontend — Auth & Shell
- [x] API client with typed wrappers (CONTRACT LOCK PROTOCOL)
- [x] Auth context with token management (localStorage)
- [x] Token refresh on 401 with auto-retry
- [x] Login page with error handling
- [x] Register page with password strength indicator
- [x] Invite accept page with all states (expired, logged in, new account)
- [x] App shell with responsive navigation
- [x] Sidebar (desktop) + mobile hamburger menu
- [x] Cycle selector component with countdown display
- [x] User menu with role badge and logout
- [x] Route guards (RequireAuth, RequireGuest, RequireLeader)
- [x] Role-based navigation (Leader Dashboard for ADMIN/SECTION_LEADER)
- [x] Placeholder pages (Dashboard, Tickets, Assignments, Practice, Leader)
- [x] All 25 frontend tests passing
- [x] TypeScript compilation passes
- [x] Mobile-responsive at all breakpoints

## Milestone 10b: Frontend — Member Dashboard + Log Practice
- [x] Install shadcn/ui components (progress, switch, accordion, dialog)
- [x] Add Dashboard and Practice Log types to types.ts
- [x] Add getMemberDashboard and createPracticeLog to API client
- [x] Member dashboard with countdown, streak, assignments, tickets
- [x] Log Practice modal with duration input and assignment selector
- [x] Blocked flag toggle with suggested ticket flow
- [x] Empty states (no assignments, no tickets, no streak)
- [x] Streak milestone celebrations (7, 14 days)
- [x] Mobile-responsive layout
- [x] 40 tests passing (1 skipped)
- [x] TypeScript compilation passes

## Milestone 10c: Frontend — Tickets UI
- [x] Ticket list page with filtering and pagination
- [x] Ticket detail/edit view
- [x] Ticket creation form
- [x] Status transition UI
- [x] Activity timeline display
- [x] Visibility and priority badges
- [x] Claim ticket functionality

## Milestone 10d: Frontend — Leader Dashboard
- [x] Leader dashboard layout
- [x] Compliance metrics display
- [x] Risk summary cards
- [x] Member drilldown views
- [x] Private ticket aggregates (privacy-safe)
- [x] Section filtering for section leaders

## Milestone 10e: Frontend — UX Polish
- [x] Loading states (page, list, button, inline)
- [x] Empty states (assignments, tickets, logs, members)
- [x] Toast notifications (success, error, warning)
- [x] Responsive behavior audit
- [x] docs/responsive.md documentation

## Milestone 11: Email Notifications & Scheduler
- [x] APScheduler dependency added and configured
- [x] Scheduler lifespan handler (startup/shutdown)
- [x] DB migration extending notification_preferences (no_log_days, weekly_digest_enabled)
- [x] Email provider abstraction (ConsoleEmailProvider, SMTPEmailProvider)
- [x] no_log_reminder job (daily)
- [x] blocking_due_48h job (every 6 hours)
- [x] blocked_over_48h job (every 6 hours)
- [x] weekly_leader_digest job (weekly)
- [x] POST /admin/jobs/{job_name}/run (ADMIN only)
- [x] GET /admin/jobs - List registered jobs
- [x] GET /teams/{team_id}/notification-preferences
- [x] PATCH /teams/{team_id}/notification-preferences
- [x] Notification preferences RBAC (users manage only their own)
- [x] Job logging (job_name, last_run, next_run, duration_ms, status)
- [x] Comprehensive tests (13 tests covering all flows)

## Milestone 12: Hardening Pass
- [x] Rate limiting on /auth/login and /auth/register (10 req/min/IP via slowapi)
- [x] Structured JSON logging with structlog
- [x] Request ID middleware (X-Request-ID header)
- [x] Enhanced /health endpoint with real DB connectivity check
- [x] RBAC & scoping audit (all endpoints verified)
- [x] Input validation audit (Pydantic layer)
- [x] Error sanitization verification (no stack traces exposed)
- [x] CORS configuration verification (env var based, no wildcards)
- [x] Index verification (all indexes from Milestone 1 exist)
- [x] N+1 query audit (leader dashboard optimized)
- [x] Pagination enforcement verification (all list endpoints)
- [x] docs/security.md (auth flow, RBAC, rate limiting, CORS)
- [x] docs/runbook.md (debugging auth, email, DB, common errors)
- [x] Rate limiting tests (in test_hardening.py)
- [x] RBAC coverage tests
- [x] Anonymization guarantee tests
- [x] Illegal status transition tests
- [x] Final CI pass (193 tests passing)
