# Legacy / Deprecated / Fallback Code Cleanup (Subagent 07)

## Search methodology

All searches were scoped to the project sources only — `backend/app/`, `frontend/src/`, `frontend/e2e/`, plus the few top-level Python/TS files. `node_modules/`, `.venv/`, `dist/`, `.pytest_cache/`, and `alembic/versions/` were excluded.

Greps run (ripgrep, case-sensitive unless noted):

- `deprecated|DEPRECATED|legacy|LEGACY|obsolete|fallback|compat|backward|backwards|shim|polyfill|kept for|previously|used to|remove this|@app\.on_event|class Config|\.dict\(\)|useHistory|<Switch>|StrictMode` against `backend/app/`.
- Same pattern (minus python-specific tokens) against `frontend/src/`.
- `TODO|FIXME|HACK|XXX` against `backend/app/` and `frontend/src/`.
- `try:[\s\S]+?except ImportError` (multiline) against `backend/app/` to catch conditional import shims.
- `query\(|session\.query|Query\(` against `backend/` to catch SQLAlchemy 1.x query style.
- `v1|v2|on_event|dict\(\)|model_dump|model_config|getattr|hasattr` against `backend/app/`.

Each hit was read in context. The vast majority of lexical hits were inside `node_modules/` / `.venv/` / `package-lock.json` and ignored. The few real hits were investigated individually and categorized.

## Findings by category

### TRUE-LEGACY (removed)

| File                                            | Snippet                                                                                                | Verdict      | Reason                                                                                                                                                                                                                                                                                                                  |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------|--------------|--|
| `backend/app/db/seed.py:356`                    | `db.query(User).filter(User.email == "manager@example.com").first()`                                  | SQLAlchemy 1.x style | The rest of the project uses `select()` (2.0 style). This is the only `db.query(...)` call in non-test backend code; converting it makes the codebase singular on 2.0 select. |
| `backend/app/tests/test_auth.py:62-64`          | `db_session.query(AuditLog).filter(AuditLog.action == AuditAction.LOGIN).all()`                       | SQLAlchemy 1.x style | Same as above. Repos & services use `select()`; these tests should match. |
| `backend/app/tests/test_permissions.py:69`      | `{log.action for log in db_session.query(AuditLog).all()}`                                            | SQLAlchemy 1.x style | Same. |
| `backend/app/tests/test_tasks_crud.py:100-104`  | `db_session.query(AuditLog).filter(AuditLog.action == AuditAction.CREATE_TASK).all()`                 | SQLAlchemy 1.x style | Same. |
| `backend/app/services/report_service.py` (×4)   | `key = status_value.value if hasattr(status_value, "value") else str(status_value)` (and 3 sibling sites for priority/decision) | Defensive `hasattr` fallback for SAEnum results | Every column read is `Task.status`, `Task.priority`, or `Approval.decision`, all declared as `Mapped[Enum]` with `SAEnum(..., values_callable=...)`. SQLAlchemy 2.0 always returns the Python enum instance for these columns. The `else str(status_value)` branch is dead and the `hasattr` shim hides what the type system already guarantees. Aligns with the task spec's "default fallback returns/values that hide misuse" bullet. |

### BUSINESS-LEGITIMATE (kept)

| File                                              | Pattern                                                                          | Reason                                                                                                                                                                                                  |
|---------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `backend/app/services/workflow_service.py:20-27`  | `VALID_TRANSITIONS` map                                                          | Explicitly called out by the task spec as a deliberate single-source-of-truth structure. Kept verbatim.                                                                                                  |
| `backend/app/db/seed.py` (whole file)             | `SEED_USERS`, `SEED_TASKS`, `_drive_status`, demo audit logs                     | Explicitly preserved by the task spec ("Seed data and demo accounts").                                                                                                                                   |
| `backend/app/api/routes/tasks.py:86,102,118,134`  | `payload: Optional[TaskActionRequest] = None` for submit/approve/reject/complete | Documented contract with the frontend: every workflow action endpoint accepts an empty body or a `{comment}` payload. Not a v1/v2 shim.                                                                  |
| `frontend/src/api/client.ts:1-5`                  | `(import.meta.env.VITE_API_BASE_URL …)?.replace(...) || DEFAULT_BASE`            | Standard env-var defaulting for local dev. Not legacy.                                                                                                                                                   |
| `frontend/src/api/client.ts:9-31`                 | `try { localStorage.* } catch { /* noop */ }`                                    | Defensive code (private-mode / SSR safety). Subagent 6's domain. Untouched.                                                                                                                              |
| `frontend/src/utils/format.ts`                    | `try { new Date(value).toLocaleString() } catch { return value }`                | Defensive code; untouched (subagent 6).                                                                                                                                                                  |
| `frontend/src/main.tsx:13-17`                     | `<StrictMode>` wrapper                                                           | Active opt-in for React strict-mode warnings, not a deprecated shim.                                                                                                                                     |
| `frontend/src/auth/AuthContext.tsx`               | exposes `error` field that has no readers                                        | Already evaluated by subagent 03 and explicitly skipped as part of the published context API.                                                                                                            |
| `backend/app/core/config.py:22-23`                | `API_V1_PREFIX`, `ENVIRONMENT` settings                                          | Already evaluated by subagent 03 and skipped (env-var contract).                                                                                                                                         |

### FALSE-POSITIVE (no action)

| File                                                                | Hit                                                                  | Why it isn't legacy                                                                                                                                                                                              |
|---------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `backend/app/main.py:23-26`                                         | "version='1.0.0'", description text                                  | This is API metadata, not a v1/v2 versioning shim. There is no `v2`.                                                                                                                                              |
| `backend/app/services/workflow_service.py:42` `assert_valid_transition` | "transition" mentioned                                            | Workflow domain word, not a "legacy → modern transition" marker.                                                                                                                                                  |
| All `model_config = ConfigDict(...)` blocks in `backend/app/schemas/` | "model_config"                                                       | Pydantic v2 idiom (intentional). Codebase has zero v1 `class Config:` blocks and zero `.dict()` calls.                                                                                                            |
| `backend/app/db/seed.py:356` `db.query(...).first()` (the seed call) | "previously..."                                                      | Re-classified as TRUE-LEGACY above (1.x style). Listed here only to note it doesn't match the "previously"/"used to" comment markers — the trigger was the SQLAlchemy 1.x query API.                              |
| `backend/.venv/...`                                                  | thousands of legacy/deprecated tokens                                | Vendored library code, out of scope per the task spec.                                                                                                                                                            |

## Parallel implementations

| Pair                                                                                                                                       | Resolution                                                                                                                                                                                                                                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------------------|---|
| `db.query(...).filter(...)` (1.x) vs `select(...).where(...)` (2.0) for the same data, in `seed.py` and three test files vs everywhere else | Removed: converted the four 1.x sites to `select()` so 2.0 is the only style.                                                                                                                                                                                                                                             |
| `value if hasattr(value, "value") else str(value)` (4 sites in `report_service.py`) vs direct `value.value` (used everywhere else, e.g. `task.status.value` in 4+ services and `assert_valid_transition`) | Removed: the `hasattr`/`str` fallback is dead under SQLAlchemy 2.0 + `SAEnum(...)` because the column always returns the Python enum. Replaced with direct `.value`.                                                                                                                                                       |

No paired old/new function names (e.g. `createXyz` vs `createXyzV2`), no duplicate route definitions, no aliased re-exports in `__init__.py` were found. The two API frontends (`api/auth.ts`, `api/tasks.ts`, etc.) each expose one canonical function per endpoint.

## Framework-version laggards

- **Pydantic**: clean. Every model uses v2 `model_config = ConfigDict(...)`; `pydantic-settings` is used for `Settings`. No `class Config:` blocks, no `.dict()` calls, no `.parse_obj(...)`, no `BaseSettings` from pydantic v1.
- **FastAPI**: clean. No `@app.on_event("startup"|"shutdown")`. The app has no startup/shutdown hooks at all (DB sessions are per-request via `Depends(get_db)`).
- **SQLAlchemy 2.0**: was mixed; now clean after this pass. All query construction in `app/repositories/`, `app/services/`, `app/db/seed.py`, and `app/tests/` uses `select(...).where(...)` after these edits.
- **React**: clean. No class components, no `componentDidMount`/`componentWillUnmount`, no `React.createClass`.
- **react-router-dom v6**: clean. Uses `Routes`/`Route`/`Navigate`/`useNavigate`/`useLocation`/`useParams`. No `Switch`, no `useHistory`, no `Redirect`.
- **No conditional `try/except ImportError` shims** anywhere in `backend/app/`.
- **No `# type: ignore` legacy markers** of substance in project sources.

## High-confidence removals (diff plan)

1. `backend/app/services/report_service.py`
   - `tasks_by_status`: replace `key = status_value.value if hasattr(status_value, "value") else str(status_value)` with `key = status_value.value`.
   - `tasks_by_priority`: replace the analogous 4-line conditional with `key = priority_value.value`.
   - `user_workload`: same simplification for the per-user `status_value`.
   - `approval_summary`: same simplification for `decision_value`.
2. `backend/app/db/seed.py`
   - Imports: add `from sqlalchemy import select` (already imported indirectly nowhere; add explicit).
   - `seed(db)`: replace `db.query(User).filter(User.email == "manager@example.com").first()` with `db.scalars(select(User).where(User.email == "manager@example.com")).first()`.
3. `backend/app/tests/test_auth.py`
   - Add `from sqlalchemy import select`.
   - Replace `db_session.query(AuditLog).filter(AuditLog.action == AuditAction.LOGIN).all()` with `list(db_session.scalars(select(AuditLog).where(AuditLog.action == AuditAction.LOGIN)).all())`.
4. `backend/app/tests/test_permissions.py`
   - Add `from sqlalchemy import select`.
   - Replace `{log.action for log in db_session.query(AuditLog).all()}` with `{log.action for log in db_session.scalars(select(AuditLog)).all()}`.
5. `backend/app/tests/test_tasks_crud.py`
   - Add `from sqlalchemy import select`.
   - Replace the `db_session.query(AuditLog).filter(...).all()` block with the `select(...).where(...)` equivalent.

No tests were deleted — every test that exercised one of the converted sites still passes against the 2.0 implementation, because SQLAlchemy 2.0 guarantees the same scalar/enum result types.

No dependencies were removed from `requirements.txt` or `package.json` — none became unused.

## Low-confidence skips

- `frontend/src/api/client.ts` `try { localStorage.* } catch { /* noop */ }` — looks like a "fallback" but is real defensive code for environments without `localStorage` (SSR, very locked-down browsers, in-memory test mocks). Subagent 6's domain.
- `frontend/src/utils/format.ts` `formatDateTime` `try/catch` — defensive parsing of an arbitrary string; subagent 6's domain.
- `frontend/src/pages/AuditLogPage.tsx:91` `logs && logs.length > 0` — still a useful empty-table guard; not a hidden `[] || something` fallback.
- `frontend/src/pages/DashboardPage.tsx:67-72` `?? 0` after `.find(...)?.count` — sane "missing-status" default for the dashboard tiles; the API can validly return fewer items than statuses if a status has zero rows. Kept.
- `frontend/src/pages/TaskListPage.tsx:89` `s.replace('_', ' ')` vs `formatStatus(s)` (used in dashboard / audit log / reports) — subtle inconsistency, but DRY consolidation is subagent 1's responsibility per the task spec, not a "legacy path" issue.
- `backend/app/main.py:21-28` `version="1.0.0"` — looks like versioning but is just OpenAPI metadata; harmless and conventional.
- `backend/app/core/security.py` `_BCRYPT_MAX_BYTES = 72` truncation — could be misread as a "shim"; it's a documented hard limit of bcrypt itself.

## Verification status

(Filled in after edits — see end of file.)

---

## Verification results (post-edit)

- **Backend `python -m py_compile` on every modified file**: clean.
- **Backend `python -m pytest -q`**: 34 passed, 0 failed (same total as the subagent 03 baseline).
- **`npx tsc --noEmit -p frontend/tsconfig.json`**: clean (no new diagnostics).
- **Frontend `npm test -- --run`**: 16 passed across 7 files.
- **`ReadLints` on every modified file**: no linter diagnostics.
- **No tests were deleted.** No dependencies were removed.
- **No framework-version migrations are recommended but undone**; the codebase is already on Pydantic v2, SQLAlchemy 2.0, FastAPI lifespan-or-no-hooks, and react-router-dom v6.
