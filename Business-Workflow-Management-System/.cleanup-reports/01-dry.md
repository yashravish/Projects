# Subagent 1 — Deduplication / DRY

## Critical assessment

The codebase is small and mostly already non-redundant — services delegate to repositories, schemas re-use enums, and pages import from a shared types module. After the prior cleanup passes (3, 7, 2, 4, 8), four duplication patterns remained where DRY *did* reduce complexity rather than just save lines:

1. **Backend `ApprovalService.approve_task` and `reject_task`** were near-identical 18-line functions differing only in the decision enum, the target status, and the default note string. The duplication risked the two paths drifting (e.g. one being patched and the other forgotten).
2. **Backend `TaskService` analyst-self check** appeared verbatim three times (`update_task`, `delete_task`, `submit_task`) — the same `if user.is_analyst and task.created_by_user_id != user.id: raise 403` block, with only the verb changing in the error string.
3. **Frontend status / priority / audit-action option arrays** were declared at the top of `TaskListPage.tsx` (statuses + priorities), `CreateTaskPage.tsx` (priorities), `EditTaskPage.tsx` (priorities), and `AuditLogPage.tsx` (actions). The literal arrays were redundant with the matching string-union types and at risk of going out of sync if the enum ever grew.
4. **Frontend "format an `unknown` error for display"** was repeated as `err instanceof Error ? err.message : 'fallback'` in six call-sites across five page files.

I did **not** consolidate these patterns (deliberately), as the abstraction would cost more than it saves:

- The four task-action methods on `TaskService` (`submit`, `approve`, `reject`, `complete`). They share a get-act-audit-commit-refresh shell, but each "act" step is a distinct call into a different collaborator (workflow, approvals, audit). Folding them into a generic helper would produce a function with five callbacks and no shared business meaning.
- Route handlers in `routes/tasks.py`. The four action endpoints look similar but each is the canonical FastAPI binding for one HTTP action; expressing them through a route factory would obscure them in the OpenAPI surface.
- `CreateTaskPage` and `EditTaskPage` form structure. They share visual layout but differ on submit logic, validation, initial-load, and disabled-state semantics. A shared `<TaskForm>` would need most fields wired through props and would not actually shrink the code.
- `UserSummary` ⇄ `UserBase` inheritance (flagged in subagent 2's report). `UserBase` is currently a request-side shape; making `UserSummary(UserBase)` blurs that role.

## High-confidence consolidations applied

### Backend

- **`backend/app/services/approval_service.py`** — extracted `_record_decision(*, task, manager, comment, decision, target_status, default_note)`. `approve_task` and `reject_task` are now three-line wrappers that pass the differing constants. Removes 16 LOC of duplicate body and locks the two paths to the same shape.
- **`backend/app/services/task_service.py`** — extracted `_ensure_creator_or_manager(task, user, *, action: str)`. `update_task`, `delete_task`, and `submit_task` now call this helper instead of repeating the same 4-line `HTTPException` block. The error message is parameterised by `action` so the messages remain accurate ("edit", "delete", "submit").

### Frontend

- **`frontend/src/types/task.ts`** — replaced the `TaskStatus` and `TaskPriority` string-union literals with `as const` arrays plus `(typeof T)[number]` types. `TASK_STATUSES` and `TASK_PRIORITIES` are now the single source of truth that both runtime iteration (option lists) and compile-time typing share.
- **`frontend/src/types/auditLog.ts`** — same treatment for `AUDIT_ACTIONS` / `AuditAction`.
- **`frontend/src/utils/format.ts`** — added `errorMessage(err: unknown, fallback: string): string`. Six call-sites updated.
- **Page updates** (`TaskListPage`, `CreateTaskPage`, `EditTaskPage`, `AuditLogPage`): import `TASK_STATUSES` / `TASK_PRIORITIES` / `AUDIT_ACTIONS` from `types/` instead of redeclaring local arrays. Local `STATUS_OPTIONS`, `PRIORITY_OPTIONS`, `ACTION_OPTIONS` constants deleted.
- **Page updates** (`TaskListPage`, `CreateTaskPage`, `EditTaskPage`, `AuditLogPage`, `DashboardPage`, `ReportsPage`, `TaskDetailPage`): replaced the `err instanceof Error ? err.message : '...'` ternary with `errorMessage(err, '...')`.

## Verification

- Backend: `python -m pytest -q` → 34 passed.
- Frontend: `npx tsc --noEmit` → clean. `npm test -- --run` → 16 passed across 7 files.
- `ReadLints` on all 12 modified files: no diagnostics.

## Low-confidence skips (intentional)

- Task-action service methods (`submit_task`, `approve_task`, `reject_task`, `complete_task`) — share the same audit/commit/refresh tail but with a distinct "do" step each. Wrapping them would require a callback parameter and harm readability.
- Route handler boilerplate (`payload.comment if payload else None`) appears 4 times in `routes/tasks.py` but is a FastAPI idiom; abstracting it would be premature.
- `CreateTaskPage` ⇄ `EditTaskPage` form duplication — a real `<TaskForm>` extraction is a worthwhile refactor but is out of scope for a DRY pass. Flagged for a future component-design pass.
- `UserSummary(UserBase)` inheritance — see `02-types.md` skip note.

## Files modified

- `backend/app/services/approval_service.py`
- `backend/app/services/task_service.py`
- `frontend/src/types/task.ts`
- `frontend/src/types/auditLog.ts`
- `frontend/src/utils/format.ts`
- `frontend/src/pages/TaskListPage.tsx`
- `frontend/src/pages/CreateTaskPage.tsx`
- `frontend/src/pages/EditTaskPage.tsx`
- `frontend/src/pages/AuditLogPage.tsx`
- `frontend/src/pages/DashboardPage.tsx`
- `frontend/src/pages/ReportsPage.tsx`
- `frontend/src/pages/TaskDetailPage.tsx`
