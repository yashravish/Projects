# Comment & Docstring Cleanup (Subagent 08)

## Summary

- Codebase comment quality is generally high; few comments are explicitly AI-slop.
- Most noise is in the form of one-line docstrings that just restate the class or function name (e.g. `class AuthService: """Encapsulates user authentication."""`).
- A handful of method docstrings in `core/security.py` and `services/audit_log_service.py` add zero information beyond the signature.
- One file (`services/task_service.py`) uses 5 large `# ----- Section -----` banners that decorate trivially-grouped methods; these add noise without informing the reader.
- A couple of in-test comments (`test_tasks_crud.py`) narrate what the next line of code obviously does.
- No commented-out code, no AI-generated/placeholder/larp markers, no stale "previously did X" history was found in non-vendored sources. All `TODO` markers in the repo live under `node_modules/` (vendored) and were left untouched.

## Categories of slop

| Category               | Count |
|------------------------|-------|
| Narration              | 2     |
| In-motion work         | 0     |
| Stub/larp              | 0     |
| Banner/section divider | 5     |
| Stale history          | 0     |
| Redundant docstring    | 7     |
| Commented-out code     | 0     |
| Self-praise            | 0     |
| Other                  | 0     |
| **Total**              | **14** |

## High-confidence removals

- `backend/app/db/base.py`
  - `class Base` body docstring `"""Declarative base for all ORM models."""` — purely restates the class name; the module docstring already says the same thing.
- `backend/app/services/auth_service.py`
  - `class AuthService` docstring `"""Encapsulates user authentication."""` — restates the name; the module docstring already covers it.
- `backend/app/services/audit_log_service.py`
  - `class AuditLogService` docstring `"""Encapsulates audit logging logic."""` — restates the name.
  - `log_action` docstring `"""Persist an audit log entry."""` — restates the function name + signature.
- `backend/app/api/deps.py`
  - `require_manager` docstring `"""Allow only manager users."""` — restates the function name.
- `backend/app/core/security.py`
  - `verify_password` docstring `"""Validate a plain password against its bcrypt hash."""` — restates the signature.
  - `create_access_token` docstring `"""Create a signed JWT for the given subject."""` — restates the signature.
- `backend/app/services/task_service.py`
  - All five `# ---- Read / Create / Update / Delete / Workflow actions ----` banner blocks (15 comment lines total) — banners over 1–4 self-named methods.
- `backend/app/tests/test_tasks_crud.py`
  - `# Create as manager` (line above `_create_task(client, manager_token, ...)`) — narrates the next line.
  - `# Other analyst (different from creator)` (line above `update = client.put(..., headers=auth_header(analyst_token))`) — narrates what the auth header already conveys; intent is captured in the test function name.

## High-confidence rewrites

None. The remaining real-content comments and docstrings (e.g. the `_BCRYPT_MAX_BYTES` rationale, the `VALID_TRANSITIONS` source-of-truth comment, `# Analysts can only edit their own tasks; managers can edit any.`, the seed module docstring, `decode_access_token`'s docstring noting it raises `JWTError`) all add real information and are kept verbatim.

## Skipped (low confidence)

- `backend/app/core/config.py` — `Settings` and `get_settings` short docstrings: technically slightly redundant but each adds a small qualifier ("backed by environment variables", "cached"). Left alone.
- `backend/app/core/security.py` — `hash_password` docstring: borderline redundant but explicitly mentions the utf-8 string return form, which is non-obvious from the type annotation. Kept.
- `backend/app/services/approval_service.py`, `report_service.py`, `task_service.py` class-level docstrings — each conveys the layer's responsibility to a new reader. Kept.
- `backend/app/repositories/*.py` class docstrings (`"Encapsulates queries against the X table."`) — boilerplate but consistent and orient the reader to the repository pattern. Kept.
- `backend/app/api/routes/*.py` module docstrings — concise route-group descriptions that match FastAPI conventions. Kept.
- `backend/app/models/*.py` `__repr__` methods, enum class docstrings — kept as-is; they're short and harmless.
- `backend/app/db/base.py` `TimestampMixin` / `IdMixin` docstrings — useful, kept.
- Frontend `// @ts-expect-error` and `/* noop */` comments in `api/client.ts` — required by tooling/lint. Kept.
- `frontend/e2e/workflow.spec.ts` `// Pick the first non-empty option in the assignee dropdown.` — explains why we use `.nth(1)`. Kept.
- All test-module docstrings (`"""Authentication tests."""` etc.) — minimal but standard pytest convention. Kept.
