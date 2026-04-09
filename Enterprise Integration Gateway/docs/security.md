# Security Considerations

## Secrets Management

- All credentials are in **environment variables only** — never committed to source control
- `.env` is in `.gitignore`; `.env.example` contains only placeholder values
- Database credentials, API keys, and URLs come from `Settings` (Pydantic BaseSettings)
- In production: use a secrets manager (HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager)

## Input Validation

- All API request parameters are validated by Pydantic schemas before they touch the database
- `skip` and `limit` query params have enforced bounds (`ge=0`, `le=500`)
- External API responses are parsed through typed Pydantic transformers, not executed as code
- XML parsing uses stdlib `ElementTree` — no external entity expansion, no network fetches

## Preventing Injection

| Attack Vector | Mitigation |
|--------------|------------|
| SQL Injection | SQLAlchemy ORM — no raw SQL string interpolation |
| XML External Entity (XXE) | Standard library `ElementTree` does not support external entities |
| JSON deserialization abuse | `json.loads` + Pydantic validation — no `eval()`, no `pickle` |
| Command injection | No shell execution in the application code |

## Error Handling

- Internal stack traces are **never** exposed in API responses
- `IntegrationError` → HTTP 502 (no internal detail leaked)
- Unhandled exceptions → HTTP 500 with generic message; full trace in server logs only
- Failed records store error messages but not sensitive credentials

## Least Privilege

- Application only needs `SELECT`, `INSERT`, `UPDATE` on its own tables
- The PostgreSQL user `eig_user` has no `DROP`, `ALTER`, or schema-level privileges in production
- Docker containers run as a non-root user (`appuser`)

## Transport Security

- In production: HTTPS only via reverse proxy (nginx / AWS ALB)
- CORS is configurable via `ALLOWED_ORIGINS` — set to specific domains in production (not `*`)

## Dependency Security

- `requirements.txt` pins exact versions to prevent supply chain drift
- Run `pip audit` or `safety check` in CI before deploying
- Base Docker image is `python:3.12-slim` (minimal attack surface)

## SDLC Practices

- GitHub Actions CI runs all tests on every push and pull request
- No secrets in CI environment — use GitHub Secrets / Actions secrets
- All new features should include corresponding tests before merge
- Code review required for changes to `transformers.py`, `sync_service.py`, and DB models

## Future Hardening (Out of Scope for v1)

- API key authentication for `/sync/*` endpoints
- Rate limiting on sync trigger endpoints (prevent abuse)
- Audit log for manual retry actions
- Field-level encryption for PII (email, phone) at rest
- Request signing for outbound HTTP to real external APIs
