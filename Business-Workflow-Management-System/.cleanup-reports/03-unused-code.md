# Unused Code Cleanup (Subagent 03)

## Tooling

Versions:

- Node `v22.22.0`, npm `11.11.0`, Python `3.12.10` (the backend's `.venv` interpreter).
- `knip` resolved by `npx --yes knip` (no version pinning, registry default).
- `ts-prune` resolved by `npx --yes ts-prune`.
- `vulture` and `pyflakes` installed into `backend/.venv` via `python -m pip install --quiet vulture pyflakes`.

Commands run (each from the noted `cwd`):

- `frontend/`: `npx --yes knip --reporter json`
- `frontend/`: `npx --yes knip` (default reporter, for human-friendly output)
- `frontend/`: `npx --yes ts-prune`
- `backend/`: `.venv/Scripts/python.exe -m vulture app --min-confidence 80`
- `backend/`: `.venv/Scripts/python.exe -m vulture app --min-confidence 60` (broader, used to surface candidates that were then individually verified)
- `backend/`: `.venv/Scripts/python.exe -m pyflakes app`

Verification commands (used after edits):

- `backend/`: `python -m py_compile <changed files>`
- `backend/`: `python -m pytest -q`
- `frontend/`: `npx tsc --noEmit -p tsconfig.json`
- `frontend/`: `npm test -- --run`
- Re-run of `vulture`, `pyflakes`, `knip`, `ts-prune` to confirm the finding count went down (or stayed the same with corresponding reductions).

No `frontend/knip.json` config was needed — knip discovered every entry on its own from `package.json` scripts and `index.html`. No new config files were created.

## Frontend findings

Raw counts (before edits):

- `knip` (default reporter): 7 unused exported types, 0 unused files, 0 unused dependencies, 0 unused devDependencies, 0 unused exports.
- `ts-prune`: 7 entries (the same 7 — every one is annotated `(used in module)`, i.e. the symbol is referenced inside its own file but not re-imported elsewhere).

Deduplicated table:

| Symbol                  | File                       | External refs | Same-file refs | Verdict           |
|-------------------------|----------------------------|---------------|----------------|-------------------|
| `ListTasksParams`       | `src/api/tasks.ts`         | 0             | 1 (param type) | Low-confidence skip |
| `ListAuditLogsParams`   | `src/api/auditLogs.ts`     | 0             | 1 (param type) | Low-confidence skip |
| `UserRole`              | `src/types/auth.ts`        | 0             | 2 (type alias) | Low-confidence skip |
| `StatusCount`           | `src/types/report.ts`      | 0             | 1 (in `TasksByStatusReport.items`) | Low-confidence skip |
| `PriorityCount`         | `src/types/report.ts`      | 0             | 1 (in `TasksByPriorityReport.items`) | Low-confidence skip |
| `UserWorkloadEntry`     | `src/types/report.ts`      | 0             | 1 (in `UserWorkloadReport.items`) | Low-confidence skip |
| `ApprovalSummaryEntry`  | `src/types/report.ts`      | 0             | 1 (in `ApprovalSummaryReport.items`) | Low-confidence skip |

All seven are exported types that are only referenced inside their own file as members of another exported type (e.g. `interface UserWorkloadReport { items: UserWorkloadEntry[] }`). They are part of the public type surface that callers are expected to consume; removing the `export` keyword would change the publicly-declared shape of these modules without any runtime benefit. Treated as low confidence and left as-is.

No unused frontend components, hooks, dependencies, or files were detected.

## Backend findings

Raw counts (before edits):

- `vulture` `--min-confidence 80`: 1 finding (`connection_record` unused variable, 100% confidence).
- `vulture` `--min-confidence 60`: 38 findings.
- `pyflakes`: 0 findings.

Of the 38 vulture findings at the 60% threshold, the vast majority are false positives: FastAPI route handlers (referenced via decorator), Pydantic model field declarations (`model_config`, `access_token`, `pending`, etc.), SQLAlchemy `relationship()` back-populated attributes, and a SQLAlchemy `event.listens_for` callback parameter. Each was verified with grep across the entire workspace.

Deduplicated table (real candidates):

| Symbol                                 | File                                   | External refs                          | Verdict           |
|----------------------------------------|----------------------------------------|----------------------------------------|-------------------|
| `TaskRepository.list_approvals`        | `app/repositories/task_repository.py`  | 0 (no callers anywhere)                | High-confidence remove |
| `TaskRepository.list_recent_workflow_events` | `app/repositories/task_repository.py` | 1 (only `WorkflowService.list_recent_events`, also dead) | High-confidence remove |
| `WorkflowService.list_recent_events`   | `app/services/workflow_service.py`     | 0                                      | High-confidence remove |
| `UserRepository.list_by_role`          | `app/repositories/user_repository.py`  | 0                                      | High-confidence remove |
| `ApprovalRead` (entire file)           | `app/schemas/approval.py`              | 0 (no `from app.schemas.approval` anywhere) | High-confidence remove (delete file) |
| `connection_record`                    | `app/tests/conftest.py:36`             | n/a — required positional arg of SQLAlchemy `connect` event listener (already `# noqa: ARG001`) | Low-confidence skip |
| `Settings.API_V1_PREFIX`               | `app/core/config.py:22`                | 0 reads in code                        | Low-confidence skip |
| `Settings.ENVIRONMENT`                 | `app/core/config.py:23`                | 0 reads in code                        | Low-confidence skip |
| All other 60%-confidence vulture hits  | various                                | False positive (FastAPI/Pydantic/SQLAlchemy) | Skip (no action) |

## High-confidence removals

Grouped by file. After each removal, dependent imports/aliases that became unused were also removed in the same edit.

- `backend/app/repositories/task_repository.py`
  - Remove method `list_approvals(self) -> Sequence[Approval]` (lines 65–67).
  - Remove method `list_recent_workflow_events(self, limit=10)` (lines 56–63).
  - Remove now-unused imports: `Sequence` (from `typing`), `Approval` (from `app.models.approval`).
- `backend/app/repositories/user_repository.py`
  - Remove method `list_by_role(self, role: UserRole)` (lines 29–31).
  - Remove now-unused import: `UserRole` (from `app.models.user`).
- `backend/app/services/workflow_service.py`
  - Remove method `list_recent_events(self, limit=10)` (lines 95–96).
- `backend/app/schemas/approval.py`
  - **Delete entire file.** Contained only `ApprovalRead`, which is referenced nowhere. No `__init__.py` re-exports needed updating (`app/schemas/__init__.py` is just a docstring).

## Low-confidence skips

- `frontend` 7 × unused exported types — referenced inside their own module as members of public response types. Removing `export` keyword is a stylistic call, not a dead-code call. Left alone.
- `app/tests/conftest.py:36` `connection_record` — required by SQLAlchemy's `event.listens_for(eng, "connect")` callback signature; already silenced with `# noqa: ARG001`. Removing it would break the listener contract.
- `app/core/config.py` `API_V1_PREFIX`, `ENVIRONMENT` — pydantic-settings fields are part of the env-driven configuration surface (`docker-compose.yml`/`.env`/operator-controlled). Even though no code reads them today, removing them silently changes the configurable env-var contract. Left as-is.
- `app/main.py` `health` / `root` — registered as FastAPI route handlers via `@app.get(...)`. False positive.
- `app/api/routes/*.py` route handler functions — registered via FastAPI decorators + `app.include_router`. False positives.
- `app/models/*.py` relationships (`workflow_events`, `assigned_tasks`, `approved_by`, etc.) — referenced via SQLAlchemy `back_populates` strings. False positives.
- `app/schemas/*.py` Pydantic field names (`access_token`, `token_type`, `model_config`, `pending`, `in_progress`, `submitted`, `approved`, `rejected`, `completed`, `approved_by_user_id`, `approved_by`, `changed_by_user_id`) — Pydantic class attributes set via constructors / used as response model fields. False positives.
- `frontend/src/auth/AuthContext.tsx` `error` field — exposed in `AuthContextValue` but no consumer destructures it; `setError(null)` is the only writer. Could be deleted, but it's part of the published context API; left alone for caller compatibility.

## Files to delete entirely

- `backend/app/schemas/approval.py` (only contained the unreferenced `ApprovalRead` class).

## Verification status

Post-edit verification (all green):

- `python -m py_compile` on every modified `.py` file: clean.
- `python -m vulture app --min-confidence 80`: 1 finding (the pre-existing `connection_record` listener parameter, low-confidence skip).
- `python -m vulture app --min-confidence 60`: dropped from 38 to 31 findings (7 reductions, exactly matching the deleted symbols: `list_approvals`, `list_by_role`, `list_recent_events`, and the 4 attributes inside the deleted `app/schemas/approval.py`).
- `python -m pyflakes app`: clean (was already clean before).
- `python -m pytest -q` (backend): **34 passed.**
- `npx tsc --noEmit -p tsconfig.json`: clean.
- `npm test -- --run` (frontend): **16 passed (7 files).**
- `npx knip` and `npx ts-prune`: unchanged from baseline (the 7 unused-export findings are the documented low-confidence skips).
- `ReadLints` on every modified file: no linter errors found.
