# Subagent 5 — Weak Types

## Methodology

Searched both sides for weak/escape-hatch typing:

- Frontend: `\b(any|unknown)\b` and `as any` / `as unknown` across `src/**/*.{ts,tsx}`.
- Backend: `\bAny\b`, `Dict[str, Any]`, `cast(`, bare `object` annotation across `backend/app/**/*.py`.
- Excluded `node_modules/`, `.venv/`, `dist/`, generated `.d.ts`.

## Findings

### Frontend — every occurrence triaged

| Site | Verdict | Reason |
| --- | --- | --- |
| `utils/format.ts` `errorMessage(err: unknown, ...)` | **Keep** — correct | Errors are spec'd as `unknown` since TS 4.4. The helper narrows internally. |
| `api/client.ts` `details?: unknown`, `body?: unknown`, `query?: Record<string, unknown>`, `let detail: unknown = undefined` | **Keep** — correct | Generic HTTP layer; request/response payloads are genuinely unknown to this module until the typed wrapper validates them. Replacing with `Record<string, unknown>` for body/query would be marginal and break the few callers that pass plain objects. |
| `pages/__tests__/*.tsx` `(fn as unknown as ReturnType<typeof vi.fn>)` (7 sites across 5 files) | **Replace** — escape hatch | Vitest 2 ships `vi.mocked(fn)` which preserves the original signature — strictly stronger typing. The double-cast was Vitest 0.x / pre-`vi.mocked` boilerplate. |

### Backend — every occurrence triaged

| Site | Verdict | Reason |
| --- | --- | --- |
| `core/security.py` `Dict[str, Any]` for `extra_claims`, `to_encode`, and `decode_access_token` return | **Replace** — concrete shape known | The token payload only ever carries a fixed set of claims (`sub`, `exp`, `iat`, `role`, `email`). Modelling it as a `TypedDict` exposes the contract to callers and removes the `Any` dependency. |
| `core/security.py` `Optional` import | Keep | Still used for `expires_minutes`. |
| `tests/conftest.py` `connection_record` (already silenced) | Keep | SQLAlchemy event-listener signature requirement. |

No `Any` appeared in models, services, repositories, or routes outside the JWT layer.

## Critical assessment

The codebase is generally strict-typed: Pydantic v2 schemas use enums and concrete types, SQLAlchemy 2.0 `Mapped[T]` annotations are everywhere, and the frontend has `strict: true` in `tsconfig.json`. The two real weaknesses were:

1. **The JWT layer leaked `Any` to its callers.** `decode_access_token` returned `Dict[str, Any]`, so `auth_service.get_user_from_token` did `payload.get("sub")` against an untyped dict and could only safely defend it with a runtime `ValueError` check.
2. **Test mocks used a type-erasing double cast (`as unknown as ReturnType<typeof vi.fn>`)** instead of the existing `vi.mocked(...)` helper. The casts threw away the original API signatures, hiding genuine type drift between fixtures and the production `Task` type. (This was confirmed when `vi.mocked` immediately surfaced a real mismatch in `TaskDetailPage.test.tsx`'s fixture, fixed in this pass.)

## High-confidence replacements applied

### Backend

- **`backend/app/core/security.py`**
  - Added `class TokenPayload(TypedDict, total=False)` with `sub`, `exp`, `iat`, `role`, `email`.
  - `decode_access_token(token: str) -> TokenPayload` (was `Dict[str, Any]`), via `cast(TokenPayload, jwt.decode(...))`.
  - `create_access_token(subject, role, email, expires_minutes=None)` — replaced the open-ended `extra_claims: Optional[Dict[str, Any]] = None` parameter with explicit, typed `role: str` / `email: str` keyword arguments. The two were the only claims any caller ever set.
  - Internal `to_encode` now typed as `TokenPayload`. `exp`/`iat` are stored as Unix integer timestamps (int, narrower than `datetime` for JWT decode/encode round-trip and matches the TypedDict).
  - Exported `TokenPayload` in `__all__`.
  - Dropped the now-unused `Any` and `Dict` imports.

- **`backend/app/services/auth_service.py`**
  - Updated the single call site of `create_access_token` to use the new explicit kwargs.

### Frontend

- **`frontend/src/pages/__tests__/TaskDetailPage.test.tsx`**: replaced two double-casts with `vi.mocked(...)`. Annotated the `submittedTask` fixture as `Task` to satisfy the now-strong return-type inference (this surfaced and fixed a real latent string-vs-literal mismatch).
- **`frontend/src/pages/__tests__/DashboardPage.test.tsx`**: four casts replaced with `vi.mocked(...)`.
- **`frontend/src/pages/__tests__/CreateTaskPage.test.tsx`**: two casts replaced.
- **`frontend/src/pages/__tests__/TaskListPage.test.tsx`**: two casts replaced.
- **`frontend/src/pages/__tests__/LoginPage.test.tsx`**: one cast replaced.

## Verification

- Backend: `python -m pytest -q` → 34 passed.
- Frontend: `npx tsc --noEmit -p tsconfig.json` → clean (and surfaced + fixed one previously-hidden type bug in `TaskDetailPage.test.tsx`).
- Frontend: `npm test -- --run` → 16 passed across 7 files.
- `ReadLints` on every modified file: no diagnostics.

## Low-confidence skips (intentional)

- `api/client.ts` generic transport types (`unknown` for body/query/details). Replacing with anything narrower (`Record<string, unknown>`) is a sideways move and breaks the boundary that the typed wrappers enforce. The current shape is correct for an HTTP client.
- `errorMessage(err: unknown, ...)` — `unknown` is the correct annotation for caught errors in TypeScript.

## Files modified

- `backend/app/core/security.py`
- `backend/app/services/auth_service.py`
- `frontend/src/pages/__tests__/TaskDetailPage.test.tsx`
- `frontend/src/pages/__tests__/DashboardPage.test.tsx`
- `frontend/src/pages/__tests__/CreateTaskPage.test.tsx`
- `frontend/src/pages/__tests__/TaskListPage.test.tsx`
- `frontend/src/pages/__tests__/LoginPage.test.tsx`
