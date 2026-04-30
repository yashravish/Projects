# Subagent 6 — Defensive Programming / try-catch Cleanup

## Methodology

Found every `try` block in the codebase (excluding `node_modules/`, `.venv/`):

- Backend: 6 sites across 5 files.
- Frontend: 21 sites across 11 files.

Each was triaged against the rule: keep only when handling genuinely unknown / unsanitized input, with clear error handling and no error hiding.

## Critical assessment — each site

### Backend

| Site | Verdict | Reason |
| --- | --- | --- |
| `app/services/auth_service.py:59` `get_user_from_token` `try: decode + int(sub) except (JWTError, ValueError, TypeError)` | **Keep** | JWT trust boundary. Tokens are external untrusted input. Catch is narrow (3 specific exceptions), re-raises as 401 with clear message. Not error-hiding. |
| `app/core/security.py:39` `verify_password` `try: bcrypt.checkpw except ValueError` | **Keep (low confidence)** | Defends against malformed bcrypt hashes in the DB. `bcrypt.checkpw` is documented to raise `ValueError` on bad hash format. Falling back to `False` means a corrupted-hash row gets a clean 401 instead of a 500. Borderline — flagged for human review. |
| `app/db/seed.py:412` `try: seed(db) finally: db.close()` | **Keep** | Resource cleanup, not exception swallowing. |
| `app/db/session.py:30` `try: yield db finally: db.close()` | **Keep** | FastAPI dependency-injection idiom for session lifecycle. |
| `app/tests/conftest.py:42, 53, 65` `try/finally` blocks | **Keep** | pytest fixture cleanup. |

### Frontend

| Site | Verdict | Reason |
| --- | --- | --- |
| `utils/format.ts:3` `formatDateTime` `try { new Date(); toLocaleString() } catch { return value }` | **Remove** | `new Date(invalid)` does not throw — it returns Invalid Date. The catch is dead code. Replaced with explicit `Number.isNaN(date.getTime())` check that actually catches malformed input. |
| `api/client.ts:10, 18, 26` `getToken/setToken/clearToken` localStorage `try/catch` | **Keep** | Real defensive code. Some browser environments throw on localStorage access (Safari "Block all cookies", iframe edge cases, quota errors). Silent fallback to null/no-op is the documented pattern; alternative is white-screen on those browsers. |
| `api/client.ts:85` nested `try { json() } catch { try { text() } catch {} }` | **Refactor (not remove)** | Logic is justified — HTTP error responses can be JSON or plaintext or empty — but the nested form is hard to read. Replaced with `.catch(() => null/'')` chains so the control flow is linear. Behaviour is identical: try JSON first, fall back to text, fall back to status-line message. |
| `pages/*` UI-action `try/catch` (DashboardPage, ReportsPage, TaskListPage, TaskDetailPage, EditTaskPage, CreateTaskPage, AuditLogPage, LoginPage — 11 sites) | **Keep** | UI/network boundary. Each catch sets a user-visible error state via `errorMessage(err, fallback)`. Clear handling, no silent swallowing. |
| `auth/AuthContext.tsx:32` bootstrap `try { fetchMe() } catch { setUser(null) }` | **Keep (low confidence)** | Auth-bootstrap-on-app-load. A 401 here is the expected "not logged in" state, not a real error. Currently swallows network errors silently too — could be tightened to differentiate `ApiError.status === 401` from real failures, but that's a behaviour decision and would interact with the unread `error` field flagged in subagent 03's report. Left alone. |

## High-confidence changes applied

- **`frontend/src/utils/format.ts`** — removed the dead `try/catch` in `formatDateTime`. Added explicit `Number.isNaN(date.getTime())` check that actually handles invalid input (the only failure mode of `new Date(string)`).
- **`frontend/src/api/client.ts`** — flattened the nested `try { response.json() } catch { try { response.text() } catch {} }` block into linear `.catch(() => null/'')` chains. Same behaviour, half the depth, no nested error swallowing. The `data` is now typed `unknown` and narrowed with an `'detail' in data` check before use.

## Verification

- Backend: `python -m pytest -q` → 34 passed.
- Frontend: `npx tsc --noEmit` → clean. `npm test -- --run` → 16 passed across 7 files.
- `ReadLints` on modified files: no diagnostics.

## Low-confidence skips (flagged for human review)

- **`security.py` `verify_password`** `except ValueError: return False`. Considered: should a malformed-hash row in the DB surface as a 500 (bug visibility) or a 401 (clean UX)? Currently 401. No change made.
- **`AuthContext.tsx` bootstrap catch**. The `setUser(null)` fallback is correct for 401 but coarsely buckets all errors (network down, DNS failure, 5xx) into "logged out". Improving requires deciding what to do with the `error` field on the context (currently set but never read by any consumer — flagged in subagents 03 and 02). No change made.
- **`api/client.ts` localStorage catches**. Silently degrades for users with localStorage disabled. Strict reading of "no fallback patterns" would suggest removing them, but that ships a worse UX for affected browsers. Kept.

## Files modified

- `frontend/src/utils/format.ts`
- `frontend/src/api/client.ts`

## What was deliberately kept

- All UI page error catches (`DashboardPage`, `ReportsPage`, `TaskListPage`, `TaskDetailPage`, `EditTaskPage`, `CreateTaskPage`, `AuditLogPage`, `LoginPage`) — these are network-boundary error displays.
- All Python `try/finally` resource-cleanup blocks (`session.py`, `seed.py`, `conftest.py`).
- `auth_service.get_user_from_token` JWT validation.
- `client.ts` localStorage browser-environment defenses.
